import torch
import torch.nn.init as init
import torch.nn as nn

from components.patching import Patch
from components.CasualAttention import CasualAttention
from components.PositionEmbedding import PositionalEncoding

class PatchTST(nn.Module):
    def __init__(self,
                 sequence_len,
                 d_model,
                 patch_size,
                 output_len,
                 beta=0.9,
                 bias=False,
                 drop_last=False,
                 device='cuda',
                 ):
        # beta is the attention connected factor
        # output_len is the length of prediction timestamp
        super(PatchTST, self).__init__()
        self.sequence_len = sequence_len
        self.d_model = d_model
        self.patch_size = patch_size
        self.output_len = output_len
        self.beta = beta
        self.drop_last = drop_last
        self.device = device
        self.num_batch = sequence_len // patch_size
        self.batch_size = self.num_batch

        if not self.drop_last:
            self.batch_size += 1

        self.Patch = Patch(
                    d_model=self.d_model,
                    sequence_len=self.sequence_len,
                    patch_size=self.patch_size,
                    device=self.device,
                    drop_last=self.drop_last
                    )

        self.embedding = PositionalEncoding(
                        d_model=self.d_model,
                        sentence_len=self.sequence_len,
                        device=self.device
                        )

        self.casualAttention1 = CasualAttention(
                                batch_size=self.batch_size,
                                sequence_len=self.patch_size,
                                d_model=self.d_model,
                                beta=self.beta,
                                bias=bias,
                                device=self.device
                                )
        self.casualAttention2 = CasualAttention(
            batch_size=self.batch_size,
            sequence_len=self.patch_size,
            d_model=self.d_model,
            beta=self.beta,
            bias=bias,
            device=self.device
        )
        self.casualAttention3 = CasualAttention(
            batch_size=self.batch_size,
            sequence_len=self.patch_size,
            d_model=self.d_model,
            beta=self.beta,
            bias=bias,
            device=self.device
        )

        self.Wo = nn.Linear(self.batch_size * self.patch_size, self.output_len, bias=bias,)
        init.xavier_normal_(self.Wo.weight, gain=nn.init.calculate_gain('linear'))
        if bias:
            init.xavier_normal_(self.Wo.bias, gain=nn.init.calculate_gain('linear'))
        self.attn = torch.zeros(size=(self.batch_size, self.patch_size, self.d_model)).to(self.device)

    def forward(self,x:torch.Tensor):
        x_sigma = torch.std(x,unbiased=False)
        norm_x = x / x_sigma
        # x:[sequence_len, d_model]
        embedded_x = self.embedding(norm_x)
        # embedded_x:[sequence_len, d_model]
        patch_x = self.Patch(embedded_x)
        # patch_x:[num_batch, patch_size, d_model]
        casual_x1, attn1 = self.casualAttention1(patch_x, self.attn)
        casual_x2, attn2 = self.casualAttention2(casual_x1, attn1)
        casual_x3, _ = self.casualAttention3(casual_x2, attn2)
        # casual_x:[patch_size, subsequence_len, d_model]
        pred_x = self.Wo(casual_x3.reshape(self.batch_size*self.patch_size, self.d_model).transpose(-1, -2)).transpose(-1, -2)
        # pred_x:[output_len, d_model]
        return pred_x * x_sigma
