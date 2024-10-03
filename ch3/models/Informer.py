import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(7, 64, 'timeF', 'h', 0.1)
        self.dec_embedding = DataEmbedding(7, 64, 'timeF', 'h', 0.1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, 3, attention_dropout=0.1,
                                      output_attention=False),
                        64, 4),
                    64,
                    128,
                    dropout=0.1,
                    activation='gelu'
                ) for l in range(2)
            ],
            [
                ConvLayer(
                    64
                ) for l in range(2 - 1)
            ] if True else None,
            norm_layer=torch.nn.LayerNorm(64)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, 3, attention_dropout=0.1, output_attention=False),
                        64, 4),
                    AttentionLayer(
                        ProbAttention(False, 3, attention_dropout=0.1, output_attention=False),
                        64, 4),
                    64,
                    128,
                    dropout=0.1,
                    activation='gelu',
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(64),
            projection=nn.Linear(64, 7, bias=True)
        )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
