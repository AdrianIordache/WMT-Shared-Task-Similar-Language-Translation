from config_file import *

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)

        pos_embedding = torch.zeros((max_len, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding      = nn.Embedding(vocab_size, embedding_size)

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
			num_encoder_layers: int,       num_decoder_layers: int, 
			embedding_size:     int,       n_head:             int, 
			src_vocab_size:     int,       tgt_vocab_size:     int, 
			dim_feedforward:    int = 512, dropout:            float = 0.1
		):

        super(Seq2SeqTransformer, self).__init__()

        self.transformer = Transformer(
			d_model            = embedding_size,
			nhead              = n_head,
			num_encoder_layers = num_encoder_layers,
			num_decoder_layers = num_decoder_layers,
			dim_feedforward    = dim_feedforward,
			dropout            = dropout
		)

        self.linear_transform    = nn.Linear(embedding_size, tgt_vocab_size)
        self.src_tok_embeddings  = TokenEmbedding(src_vocab_size, embedding_size)
        self.tgt_tok_embeddings  = TokenEmbedding(tgt_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout = dropout)


    def forward(self,
            src: Tensor,
            tgt: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor,
            memory_key_padding_mask: Tensor
    	):

        src_embedding = self.positional_encoding(self.src_tok_embeddings(src))
        tgt_embedding = self.positional_encoding(self.tgt_tok_embeddings(tgt))

        outs = self.transformer(
			src                     = src_embedding,  
			tgt                     = tgt_embedding, 
			src_mask                = src_mask,       
			tgt_mask                = tgt_mask, 
			memory_mask             = None,
			src_key_padding_mask    = src_padding_mask, 
			tgt_key_padding_mask    = tgt_padding_mask, 
			memory_key_padding_mask = memory_key_padding_mask
       )

        return self.linear_transform(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
						self.positional_encoding(self.src_tok_embeddings(src)), 
						src_mask
					)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
						self.positional_encoding(self.tgt_tok_embeddings(tgt)), 
						memory, 
						tgt_mask
					)

    
class EncoderRNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class AttentionRNN(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class DecoderRNN(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2SeqRNN(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask                = None,       
                tgt_mask                = None, 
                memory_mask             = None,
                src_key_padding_mask    = None, 
                tgt_key_padding_mask    = None, 
                memory_key_padding_mask = None,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs

class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.
    """

    def __init__(self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors)
        :param d_values: size of value vectors
        :param dropout: dropout probability
        :param in_decoder: is this Multi-Head Attention sublayer instance in the decoder?
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries  # size of key vectors, same as of the query vectors to allow dot-products for similarity

        self.in_decoder = in_decoder

        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))

        # A linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors (of the same size as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        """
        Forward prop.

        :param query_sequences: the input query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        :param key_value_sequences: the sequences to be queried against, a tensor of size (N, key_value_sequence_pad_length, d_model)
        :param key_value_sequence_lengths: true lengths of the key_value_sequences, to be able to ignore pads, a tensor of size (N)
        :return: attention-weighted output sequences for the query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        """
        batch_size = query_sequences.size(0)  # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_value_sequences, query_sequences)

        # Store input for adding later
        input_to_add = query_sequences.clone()

        # Apply layer normalization
        query_sequences = self.layer_norm(query_sequences)  # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value sequences (as they are the same as the query sequences)
        # If this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self_attention:
            key_value_sequences = self.layer_norm(key_value_sequences)  # (N, key_value_sequence_pad_length, d_model)

        # Project input sequences to queries, keys, values
        queries = self.cast_queries(query_sequences)  # (N, query_sequence_pad_length, n_heads * d_queries)
        keys, values = self.cast_keys_values(key_value_sequences).split(split_size=self.n_heads * self.d_keys,
                                                                        dim=-1)  # (N, key_value_sequence_pad_length, n_heads * d_keys), (N, key_value_sequence_pad_length, n_heads * d_values)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.n_heads,
                                            self.d_queries)  # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads,
                                      self.d_keys)  # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads,
                                          self.d_values)  # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        # And then, for convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
        queries = queries.permute(0, 2, 1, 3).contiguous().view(-1, query_sequence_pad_length,
                                                                self.d_queries)  # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = keys.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_length,
                                                          self.d_keys)  # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = values.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_length,
                                                              self.d_values)  # (N * n_heads, key_value_sequence_pad_length, d_values)

        # Perform multi-head attention

        # Perform dot-products
        attention_weights = torch.bmm(queries, keys.permute(0, 2,
                                                            1))  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (1. / math.sqrt(
            self.d_keys)) * attention_weights  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(
            attention_weights).to(DEVICE)  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.n_heads).unsqueeze(
            1).unsqueeze(2).expand_as(
            attention_weights)  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float(
            'inf'))  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the decoder, keys chronologically ahead of queries
        if self.in_decoder and self_attention:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
            not_future_mask = torch.ones_like(
                attention_weights).tril().bool().to(
                DEVICE)  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(~not_future_mask, -float(
                'inf'))  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Compute softmax along the key dimension
        attention_weights = self.softmax(
            attention_weights)  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Apply dropout
        attention_weights = self.apply_dropout(
            attention_weights)  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values)  # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original order of axes
        sequences = sequences.contiguous().view(batch_size, self.n_heads, query_sequence_pad_length,
                                                self.d_values).permute(0, 2, 1,
                                                                       3)  # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length,
                                                -1)  # (N, query_sequence_pad_length, n_heads * d_values)

        # Transform the concatenated subspace-sequences into a single output of size d_model
        sequences = self.cast_output(sequences)  # (N, query_sequence_pad_length, d_model)

        # Apply dropout and residual connection
        sequences = self.apply_dropout(sequences) + input_to_add  # (N, query_sequence_pad_length, d_model)

        return sequences


class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, d_model, d_inner, dropout):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # A linear layer to project from the input size to an intermediate size
        self.fc1 = nn.Linear(d_model, d_inner)

        # ReLU
        self.relu = nn.ReLU()

        # A linear layer to project from the intermediate size to the output size (same as the input size)
        self.fc2 = nn.Linear(d_inner, d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward prop.

        :param sequences: input sequences, a tensor of size (N, pad_length, d_model)
        :return: transformed output sequences, a tensor of size (N, pad_length, d_model)
        """
        # Store input for adding later
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        # Apply layer-norm
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Transform position-wise
        sequences = self.apply_dropout(self.relu(self.fc1(sequences)))  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)

        # Apply dropout and residual connection
        sequences = self.apply_dropout(sequences) + input_to_add  # (N, pad_length, d_model)

        return sequences


class Encoder(nn.Module):
    """
    The Encoder.
    """

    def __init__(self, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers,
                 dropout):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for the Encoder
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_layers: number of [multi-head attention + position-wise FC] layers in the Encoder
        :param dropout: dropout probability
        """
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # An embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Set the positional encoding tensor to be un-update-able, i.e. gradients aren't computed
        self.positional_encoding.requires_grad = False

        # Encoder layers
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer() for i in range(n_layers)])

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        """
        Creates a single layer in the Encoder by combining a multi-head attention sublayer and a position-wise FC sublayer.
        """
        # A ModuleList of sublayers
        encoder_layer = nn.ModuleList([MultiHeadAttention(d_model=self.d_model,
                                                          n_heads=self.n_heads,
                                                          d_queries=self.d_queries,
                                                          d_values=self.d_values,
                                                          dropout=self.dropout,
                                                          in_decoder=False),
                                       PositionWiseFCNetwork(d_model=self.d_model,
                                                             d_inner=self.d_inner,
                                                             dropout=self.dropout)])

        return encoder_layer

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        """
        Forward prop.

        :param encoder_sequences: the source language sequences, a tensor of size (N, pad_length)
        :param encoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :return: encoded source language sequences, a tensor of size (N, pad_length, d_model)
        """
        pad_length = encoder_sequences.size(1)  # pad-length of this batch only, varies across batches

        # Sum vocab embeddings and position embeddings
        encoder_sequences = self.embedding(encoder_sequences) * math.sqrt(self.d_model) + self.positional_encoding[:,
                                                                                          :pad_length, :].to(
            DEVICE)  # (N, pad_length, d_model)

        # Dropout
        encoder_sequences = self.apply_dropout(encoder_sequences)  # (N, pad_length, d_model)

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            # Sublayers
            encoder_sequences = encoder_layer[0](query_sequences=encoder_sequences,
                                                 key_value_sequences=encoder_sequences,
                                                 key_value_sequence_lengths=encoder_sequence_lengths)  # (N, pad_length, d_model)
            encoder_sequences = encoder_layer[1](sequences=encoder_sequences)  # (N, pad_length, d_model)

        # Apply layer-norm
        encoder_sequences = self.layer_norm(encoder_sequences)  # (N, pad_length, d_model)

        return encoder_sequences


class Decoder(nn.Module):
    """
    The Decoder.
    """

    def __init__(self, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers,
                 dropout):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for the Decoder
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_layers: number of [multi-head attention + multi-head attention + position-wise FC] layers in the Decoder
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # An embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Set the positional encoding tensor to be un-update-able, i.e. gradients aren't computed
        self.positional_encoding.requires_grad = False

        # Decoder layers
        self.decoder_layers = nn.ModuleList([self.make_decoder_layer() for i in range(n_layers)])

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Output linear layer that will compute logits for the vocabulary
        self.fc = nn.Linear(d_model, vocab_size)

    def make_decoder_layer(self):
        """
        Creates a single layer in the Decoder by combining two multi-head attention sublayers and a position-wise FC sublayer.
        """
        # A ModuleList of sublayers
        decoder_layer = nn.ModuleList([MultiHeadAttention(d_model=self.d_model,
                                                          n_heads=self.n_heads,
                                                          d_queries=self.d_queries,
                                                          d_values=self.d_values,
                                                          dropout=self.dropout,
                                                          in_decoder=True),
                                       MultiHeadAttention(d_model=self.d_model,
                                                          n_heads=self.n_heads,
                                                          d_queries=self.d_queries,
                                                          d_values=self.d_values,
                                                          dropout=self.dropout,
                                                          in_decoder=True),
                                       PositionWiseFCNetwork(d_model=self.d_model,
                                                             d_inner=self.d_inner,
                                                             dropout=self.dropout)])

        return decoder_layer

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        """
        Forward prop.

        :param decoder_sequences: the source language sequences, a tensor of size (N, pad_length)
        :param decoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :param encoder_sequences: encoded source language sequences, a tensor of size (N, encoder_pad_length, d_model)
        :param encoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :return: decoded target language sequences, a tensor of size (N, pad_length, vocab_size)
        """
        pad_length = decoder_sequences.size(1)  # pad-length of this batch only, varies across batches

        # Sum vocab embeddings and position embeddings
        decoder_sequences = self.embedding(decoder_sequences) * math.sqrt(self.d_model) + self.positional_encoding[:,
                                                                                          :pad_length, :].to(
            DEVICE)  # (N, pad_length, d_model)

        # Dropout
        decoder_sequences = self.apply_dropout(decoder_sequences)

        # Decoder layers
        for decoder_layer in self.decoder_layers:
            # Sublayers
            decoder_sequences = decoder_layer[0](query_sequences=decoder_sequences,
                                                 key_value_sequences=decoder_sequences,
                                                 key_value_sequence_lengths=decoder_sequence_lengths)  # (N, pad_length, d_model)
            decoder_sequences = decoder_layer[1](query_sequences=decoder_sequences,
                                                 key_value_sequences=encoder_sequences,
                                                 key_value_sequence_lengths=encoder_sequence_lengths)  # (N, pad_length, d_model)
            decoder_sequences = decoder_layer[2](sequences=decoder_sequences)  # (N, pad_length, d_model)

        # Apply layer-norm
        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)

        # Find logits over vocabulary
        decoder_sequences = self.fc(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences


class Transformer(nn.Module):

    def __init__(self, vocab_size: int, max_seq_len: int = 256, n_heads: int = 8,
        d_model: int = 512, d_queries: int = 64, d_values: int = 64, d_feed_forward: int = 2048, 
        n_layers: int = 6, dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        self.vocab_size     = vocab_size
        self.max_seq_len    = max_seq_len
        self.n_heads        = n_heads
        self.d_model        = d_model
        self.d_queries      = d_queries
        self.d_values       = d_values
        self.d_feed_forward = d_feed_forward
        self.n_layers       = n_layers
        self.dropout        = dropout

        self.positional_encoding = get_positional_encoding(
            d_model     = d_model, 
            max_seq_len = max_seq_len
        )

        self.encoder = Encoder(
            vocab_size          = self.vocab_size,
            positional_encoding = self.positional_encoding,
            d_model             = self.d_model,
            n_heads             = self.n_heads,
            d_queries           = self.d_queries,
            d_values            = self.d_values,
            d_inner             = self.d_feed_forward,
            n_layers            = self.n_layers,
            dropout             = self.dropout
        )

        self.decoder = Decoder(
            vocab_size          = self.vocab_size,
            positional_encoding = self.positional_encoding,
            d_model             = self.d_model,
            n_heads             = self.n_heads,
            d_queries           = self.d_queries,
            d_values            = self.d_values,
            d_inner             = self.d_feed_forward,
            n_layers            = self.n_layers,
            dropout             = self.dropout
        )

        self.init_weights()

    def init_weights(self):
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain = 1.)

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(self.encoder.embedding.weight, mean = 0., std = math.pow(self.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.decoder.fc.weight        = self.decoder.embedding.weight

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        """
        :param encoder_sequences: source language sequences, a tensor of size (N, encoder_sequence_pad_length)
        :param decoder_sequences: target language sequences, a tensor of size (N, decoder_sequence_pad_length)
        :param encoder_sequence_lengths: true lengths of source language sequences, a tensor of size (N)
        :param decoder_sequence_lengths: true lengths of target language sequences, a tensor of size (N)
        :return: decoded target language sequences, a tensor of size (N, decoder_sequence_pad_length, vocab_size)
        """

        encoder_sequences = self.encoder(encoder_sequences,
                                         encoder_sequence_lengths)  # (N, encoder_sequence_pad_length, d_model)

        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences,
                                         encoder_sequence_lengths)  # (N, decoder_sequence_pad_length, vocab_size)

        return decoder_sequences

def get_model(CFG: Dict) -> nn.Module:
    assert CFG['architecture_type'].lower() in ['transformer', 'rnn'], \
        'Invalid architecture type. Options are: Transformer / RNN'

    if CFG['architecture_type'].lower() == 'transformer':
        return Transformer(
            vocab_size     = CFG['vocab_size'],
            max_seq_len    = CFG['max_seq_len'],
            n_heads        = CFG['n_heads'],
            d_model        = CFG['d_model'],
            d_queries      = CFG['d_queries'],
            d_values       = CFG['d_values'],
            d_feed_forward = CFG['d_feed_forward'],
            n_layers       = CFG['n_layers'],
            dropout        = CFG['dropout']
        )
    
    if CFG['architecture_type'].lower() == 'rnn':
        encoder   = EncoderRNN(
            CFG['src_vocab_size'], 
            CFG['embedding_size'], 
            CFG['encoder_hidden_dim'], 
            CFG['decoder_hidden_dim'], 
            CFG['encoder_dropout']
        )
        
        attention = AttentionRNN(
            CFG['encoder_hidden_dim'], 
            CFG['decoder_hidden_dim'], 
            CFG['attention_dim']
        )
        
        decoder   = DecoderRNN(
            CFG['tgt_vocab_size'], 
            CFG['embedding_size'], 
            CFG['encoder_hidden_dim'], 
            CFG['decoder_hidden_dim'], 
            CFG['decoder_dropout'], 
            attention
        )
        
        return Seq2SeqRNN(encoder, decoder, DEVICE)