from __future__ import print_function
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Semantic_Match(nn.Module):
    def __init__(self, bit):
        super(Semantic_Match, self).__init__()

        self.X_dim = bit
        self.C_dim = 275
        self.vae_enc_drop = 0.4
        self.block_dim = 64
        self.channel = self.C_dim
        self.reduction = 16
        # 首先进行分好块
        self.blocks = nn.Sequential(
            nn.Linear(self.X_dim, self.X_dim),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.X_dim, self.C_dim*self.block_dim),
        )
        # 执行语义注意力模块
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.SE_attention = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid()
        )


    def attention(self,x):
        x = x.view(x.shape[0],self.C_dim,int(math.sqrt(self.block_dim)),int(math.sqrt(self.block_dim)))
        b,c,h,w = x.size()
        y = self.avgPool(x).view(b,c)
        y = self.SE_attention(y).view(b,c,1,1)
        z = x * y.expand_as(x)
        return z.view(x.shape[0],self.C_dim,-1)

    def forward(self, x):
        # 分块
        x_block = self.blocks(x).view(x.shape[0],self.C_dim,self.block_dim)
        # 注意力
        x_attention = self.attention(x_block)

        return F.sigmoid(x_attention)






