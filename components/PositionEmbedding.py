import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, sentence_len, device='cuda'):
        # d_model是词向量维度,
        # sentence_len是句子最大长度
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(sentence_len, d_model).to(device)
        sin_position = torch.arange(0, sentence_len, 2).unsqueeze(1).to(device)
        cos_position = torch.arange(1, sentence_len, 2).unsqueeze(1).to(device)
        div_term = torch.exp(
            torch.arange(0, d_model) * -(math.log(10000.0) / d_model)
        ).to(device)
        pe[0::2, :] = torch.sin(sin_position * div_term).to(device)
        pe[1::2, :] = torch.cos(cos_position * div_term).to(device)
        #unsqueeze函数用于增加的方括号保证数组的维度一致
        self.pe = pe

    def forward(self, x):
        x = x + self.pe[:x.size(0), : x.size(1)].requires_grad_(False)
        return x
