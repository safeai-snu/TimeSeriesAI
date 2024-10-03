import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead


from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 7
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (2 - 1)))
        self.head_nf = 64 * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(64, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, 7, self.in_seg_num,64))
        self.pre_norm = nn.LayerNorm(64)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, 64, 4, 128,
                            1, 0.1,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l),3
                            ) for l in range(2)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, 7, (self.pad_out_len // self.seg_len), 64))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), 3, 64, 4,
                                           128, 0.1),
                    AttentionLayer(
                        FullAttention(False, 3, attention_dropout=0.1,
                                      output_attention=False),
                        64,4),
                    self.seg_len,
                    64,
                    128,
                    dropout=0.1,
                    # activation='gelu',
                )
                for l in range(2 + 1)
            ],
        )



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
