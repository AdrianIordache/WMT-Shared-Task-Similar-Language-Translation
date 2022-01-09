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