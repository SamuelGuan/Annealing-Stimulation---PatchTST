import torch
import torch.nn.init as init
import torch.nn as nn

class CasualAttention(nn.Module):
    '''
    :param batch_size : how many sequences to be inputed
    :param sequence_len: the number of words in one batch
    :param d_model: word-embedding dimension
    :param beta moving average factor between attn in this casual attention block
           and attn in casual attention lask block
    '''

    def __init__(self,
                 batch_size,
                 sequence_len,
                 d_model,
                 beta=0.9,
                 device='cuda',
                 bias = False,
                 ):
        super(CasualAttention, self).__init__()
        self.device = device
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.d_model = d_model
        self.beta = beta
        self.bias = bias

        # define learnable parameters
        self.Wq = nn.Linear(sequence_len, sequence_len, bias=bias)
        self.Wk = nn.Linear(sequence_len, sequence_len, bias=bias)
        self.Wv = nn.Linear(sequence_len, sequence_len, bias=bias)
        self.Wh = nn.Linear(d_model, d_model, bias=bias)
        self.FFN = nn.Linear(d_model, d_model, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

        self.gelu = nn.GELU()
        self.beta = beta

        #权重初始化
        self.weight_initiate()

    def forward(self, x:torch.Tensor, last_attn:torch.Tensor):
        # origin_x:[batch_size,sequence_len,d_model]
        origin_x = x
        # instanceNorm
        sigma = torch.var(x, dim=-1, keepdim=True)
        x = x * torch.sqrt(1/(sigma + 0.001))
        # x:[batch_size,sequence_len,d_model]
        x = x.transpose(-1,-2)
        # x:[batch_size,d_model,sequence_len]
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x).transpose(-1,-2)
        # q,k:[batch_size,d_model,sequence_len]
        # v:[batch_size, sequence_len, d_model]
        scores = torch.sqrt(torch.tensor(1/self.sequence_len)) * (q.transpose(-1,-2) @ k)
        # scores:[batch_size, sequence_len, sequence_len]
        # mask the scores matrix
        scores_with_mask = self.softmax((scores).transpose(-1,-2)).transpose(-1,-2)
        # scores_with_mask:[batch_size, sequence_len, sequence_len]
        chips = scores_with_mask @ v
        # chips:[batch_size, sequence_len, d_model]
        # flatten as concatenate function
        chip = chips.reshape(-1, self.d_model)
        # chip:[batch_size*sequence_len, d_model]

        origin_attn = (self.beta*self.Wh(chip)).reshape(
                                            [self.batch_size,
                                            self.sequence_len,
                                            self.d_model]
                                            ) +(1-self.beta)*last_attn
        attn = origin_attn +origin_x
        # attn:[bathc_size, sequence_len, d_model]
        attn_sigma = torch.var(attn, dim=-1, keepdim=True)
        # instanceNorm
        norm_attn = attn * torch.sqrt(1/(attn_sigma + 0.001))
        # FFN
        return self.gelu(self.FFN(norm_attn)+attn), origin_attn

    def weight_initiate(self):
        self.Wq.weight.data.normal_(0, 1 / self.sequence_len)
        self.Wv.weight.data.normal_(0, 1 / self.sequence_len)
        self.Wk.weight.data.normal_(0, 1 / self.sequence_len)
        init.xavier_normal_(self.Wh.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            self.Wq.bias.data.normal_(0, 1 / self.sequence_len)
            self.Wv.bias.data.normal_(0, 1 / self.sequence_len)
            self.Wk.bias.data.normal_(0, 1 / self.sequence_len)
            init.xavier_normal_(self.Wh.bias, gain=nn.init.calculate_gain('relu'))

