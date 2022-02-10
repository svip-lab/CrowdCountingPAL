from __future__ import absolute_import, print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model_vq import Quantize
from model_csrnet import CSRNet as net_csr

import utils
import ipdb
args = utils.parse_command()


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # VQ
        context_dim = 512
        embed_dim = 64
        n_embed = 512
        self.quantize = Quantize(embed_dim, n_embed)

        channel = 512
        context_dim = 512 + channel
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_deconv = nn.Conv2d(embed_dim, channel, 1)

        self.csrnet_1 = net_csr()
        self.csrnet_2 = net_csr()

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in xrange(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

            # import pdb;pdb.set_trace()
        self.upsampling = nn.Upsample(scale_factor=8, mode='nearest')
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x, mask=None, target=None, train_flag=True):
        diff_mean, diff_var = None, None

        x = self.frontend(x)
        features = x
        x_dis = x.clone()

        if target is not None:
            target = (target - target.min() )/ (target.max() - target.min())
            x_dis = x_dis * target

        if mask is not None:
            diff_mean, diff_var=[], []

            x_dis = x_dis.permute(0,2,3,1)
            x_ann = x_dis[mask.bool()]
            x_unkn = x_dis[mask.bool()==0]

            x_ann_mean, x_ann_var = torch.var_mean(x_ann, dim=0)
            x_unkn_mean, x_unkn_var = torch.var_mean(x_unkn, dim=0)

            x_ann_ = x_ann-x_ann_mean
            x_unkn_ = x_unkn-x_unkn_mean
            x_ann_var = torch.matmul(x_ann_.permute(1,0), x_ann_) / x_ann_.shape[0]
            x_unkn_var = torch.matmul( x_unkn_.permute(1,0), x_unkn_ ) / x_unkn_.shape[0]

            diff_mean.append(x_ann_mean)
            diff_mean.append(x_unkn_mean)

            diff_var.append(x_ann_var)
            diff_var.append(x_unkn_var)


        quant = self.quantize_conv(x).permute(0, 2, 3, 1)
        quant, diff, id_t = self.quantize(quant, mask, train_flag)
        quant = quant.permute(0, 3, 1, 2)
        quant = self.quantize_deconv(quant)

        x = torch.cat([x, quant], dim=1)

        output_csrnet_1 = self.csrnet_1(x)
        output_csrnet_2 = self.csrnet_2(x)

        return output_csrnet_1, output_csrnet_2, diff, id_t, diff_mean, diff_var

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

