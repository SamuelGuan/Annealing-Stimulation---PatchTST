import torch
import math
import torch.nn as nn

class Patch(nn.Module):
    def __init__(self,
                 d_model,
                 sequence_len,
                 patch_size,
                 device='cuda',
                 drop_last=True
                 ):
        super(Patch, self).__init__()
        self.patch_size = patch_size
        self.drop_last = drop_last
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.num_batch = sequence_len // patch_size

        if not self.drop_last:
            self.zeros = torch.zeros(size=(patch_size - sequence_len%patch_size,d_model)).to(device)

    def forward(self,x:torch.Tensor):
        #x:[sequence_len, d_model]
        if not self.drop_last:
            x = torch.concat([self.zeros,x],dim=0)
            x = x.reshape(self.num_batch+1, self.patch_size, self.d_model)
        else:
            x = x.reshape(self.num_batch, self.patch_size, self.d_model)

        return x.requires_grad_(False)

    def get_num_batch(self):
        return self.num_batch