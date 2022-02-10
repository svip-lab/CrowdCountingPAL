import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
import ipdb

import distributed as dist_fn
import utils
args = utils.parse_command()

## 2020-11-03 10:00 XUYY add VQ
## Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch
## https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L216

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super(Quantize, self).__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, mask=None, train_flag=True):
        flatten = input.reshape(-1, self.dim)
        #import pdb; pdb.set_trace()
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(flatten, self.embed) 
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # 2 * flatten @ self.embed

        #import pdb;pdb.set_trace()
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if mask is not None:
            #import pdb;pdb.set_trace()
            input_new = input[mask.bool()]
            flatten_new = input_new#.reshape(-1, self.dim)
            dist_new = (
                flatten_new.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(flatten_new, self.embed)
                + self.embed.pow(2).sum(0, keepdim=True)
            )

            _, embed_ind_new = (-dist_new).max(1)
            embed_onehot_new=F.one_hot(embed_ind_new,self.n_embed).type(flatten_new.dtype)
            #embed_ind = embed_ind.view(*input.shape[:-1])


        if self.training and train_flag:
            
            embed_onehot_sum = embed_onehot_new.sum(0)
            #embed_sum = flatten.transpose(0, 1) @ embed_onehot
            embed_sum = torch.matmul(flatten_new.transpose(0, 1), embed_onehot_new)

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))



