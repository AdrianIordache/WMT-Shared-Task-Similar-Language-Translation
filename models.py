from utils import *

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


def get_model(CFG: Dict) -> nn.Module:
    assert CFG['architecture_type'].lower() in ['transformer', 'rnn'], \
        'Invalid architecture type. Options are: Transformer / RNN'

    if CFG['architecture_type'].lower() == 'transformer':
        return Seq2SeqTransformer(
            CFG['num_encoder_layers'], 
            CFG['num_decoder_layers'], 
            CFG['embedding_size'], 
            CFG['n_heads'], 
            CFG['src_vocab_size'], 
            CFG['tgt_vocab_size'], 
            CFG['ffn_hidden_dim']
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