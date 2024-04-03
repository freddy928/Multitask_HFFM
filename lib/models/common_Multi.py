import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F

from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Conv_decoder
from torch.nn import Upsample


# channel + self attention
class qkvGateAttention(nn.Module):

    def __init__(self, ch1=512, ch2=256, upsamsize=2, isup=True, nhead=1, N=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # gap channel attention
        # qkv_bias, attn_drop, proj_drop: self-attention parameter

        self.nhead = nhead
        self.gap= nn.AdaptiveAvgPool2d((N,1))
        dim= ch1+ ch2
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooling= nn.Conv1d(in_channels=N,out_channels=1,kernel_size=1,stride=1)

        # upsample out or downsample out
        self.isup= isup
        if isup== True:
            self.upConv= nn.Sequential(
                Conv(ch1, ch2, 1, 1),
                Upsample(None, upsamsize, 'nearest')
            )
            self.bottle= BottleneckCSP(ch2* 2, ch2, 1, False)
        else:
            self.downConv= Conv(ch2, ch1, 3, upsamsize)
            self.bottle= BottleneckCSP(ch1* 2, ch1* 2, 1, False)

    def forward(self, input):
        feat1, feat2= input  # feat1:b 4c h/2 w/2  feat2:b c h w
        b, c1, _, _= feat1.shape
        b, c2, _, _= feat2.shape

        feat1_gap= self.gap(feat1).view(b,c1,-1)
        feat2_gap= self.gap(feat2).view(b,c2,-1)
        feat_token= torch.cat([feat1_gap, feat2_gap], dim=1).permute(0,2,1)   # b n c
        # self attention dimension
        B, N, C= feat_token.shape

        qkv= self.qkv(feat_token).reshape(B, N, 3, self.nhead, C// self.nhead).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, C//num_heads]

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        featout = (attn @ v).transpose(1, 2).reshape(B, N, C)

        featout = self.proj(featout)
        featout = self.proj_drop(featout)

        feat_wgh= self.pooling(featout).squeeze(1)

        feat_ch1 = feat1 * feat_wgh[:, 0:c1].view(b, c1, 1, 1)
        feat_ch2 = feat2 * feat_wgh[:, c1:c1 + c2].view(b, c2, 1, 1)

        # add after
        if self.isup == True:
            feat_h_up = self.upConv(feat_ch1)
            feat = torch.cat([feat_h_up, feat_ch2], dim=1)
            feat = self.bottle(feat)
        else:
            feat_l_down = self.downConv(feat_ch2)
            feat = torch.cat([feat_l_down, feat_ch1], dim=1)
            feat = self.bottle(feat)
        return feat

# channel + mixed attention
class qkvGateAttention2(nn.Module):

    def __init__(self, ch1=512, ch2=256, upsamsize=2, isup=True, nhead=1, N=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # qkv_bias, attn_drop, proj_drop: self-attention parameter

        self.nhead= nhead
        self.ch1 = ch1
        self.ch2 = ch2
        self.gap= nn.AdaptiveAvgPool2d((N* nhead,1))
        dim= ch1+ ch2
        head_dim = dim // nhead

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooling= nn.Conv1d(in_channels=N* nhead,out_channels=1,kernel_size=1,stride=1)

        # upsample out or downsample out
        self.isup= isup
        if isup== True:
            self.upConv= nn.Sequential(
                Conv(ch1, ch2, 1, 1),
                Upsample(None, upsamsize, 'nearest')
            )
            self.bottle= BottleneckCSP(ch2* 2, ch2, 1, False)
        else:
            self.downConv= Conv(ch2, ch1, 3, upsamsize)
            self.bottle= BottleneckCSP(ch1* 2, ch1* 2, 1, False)

    def forward(self, input):
        feat1, feat2= input  # feat1:b 4c h/2 w/2  feat2:b c h w
        b, c1, _, _= feat1.shape
        b, c2, _, _= feat2.shape

        feat1_gap= self.gap(feat1).view(b,c1,-1)
        feat2_gap= self.gap(feat2).view(b,c2,-1)
        feat_token= torch.cat([feat1_gap, feat2_gap], dim=1).permute(0,2,1)   # b n c
        # self attention dimension
        B, N, C= feat_token.shape

        qkv= self.qkv(feat_token).reshape(B, N// self.nhead, 3, self.nhead, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, C//num_heads]

        # channel + mixed attention
        q_ch1, q_ch2= torch.split(q, [self.ch1, self.ch2], dim=-1)
        k_ch1, k_ch2= torch.split(k, [self.ch1, self.ch2], dim=-1)
        v_ch1, v_ch2= torch.split(v, [self.ch1, self.ch2], dim=-1)

        attn = (q_ch1.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_ch1 = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B, N, self.ch1)

        attn = (q_ch2.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_ch2 = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B, N, self.ch2)

        featout = torch.cat([x_ch1, x_ch2], dim=-1)
        featout = self.proj(featout)
        featout = self.proj_drop(featout)

        feat_wgh= self.pooling(featout).squeeze(1)

        feat_ch1 = feat1 * feat_wgh[:, 0:c1].view(b, c1, 1, 1)
        feat_ch2 = feat2 * feat_wgh[:, c1:c1 + c2].view(b, c2, 1, 1)

        # add after
        if self.isup == True:
            feat_h_up = self.upConv(feat_ch1)
            feat = torch.cat([feat_h_up, feat_ch2], dim=1)
            feat = self.bottle(feat)
        else:
            feat_l_down = self.downConv(feat_ch2)
            feat = torch.cat([feat_l_down, feat_ch1], dim=1)
            feat = self.bottle(feat)
        return feat


class gateAttention(nn.Module):
    # gate attention
    def __init__(self, ch_in=512, ch_out=256, upsamsize=4, nhead=3, isup=True):
        super().__init__()
        # gap channel attention
        self.nhead= nhead
        self.gap= nn.AdaptiveAvgPool2d((nhead,1))
        self.wghLinear= nn.Sequential(
            nn.Conv1d(in_channels=ch_in + ch_out, out_channels=(ch_in+ch_out)// 2, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=(ch_in + ch_out)// 2, out_channels=ch_in + ch_out, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.wghLinear2= nn.Linear(in_features=(ch_in+ ch_out)* nhead,out_features=ch_in+ ch_out)
        self.isup= isup
        if isup== True:
            self.upConv= nn.Sequential(
                Conv(ch_in, ch_out, 1, 1),
                Upsample(None, upsamsize, 'nearest')
            )
            self.bottle= BottleneckCSP(ch_out* 2, ch_out, 1, False)
        else:
            self.downConv= Conv(ch_out, ch_in, 3, upsamsize)
            self.bottle= BottleneckCSP(ch_in* 2, ch_in* 2, 1, False)

    def forward(self,input):

        feat_h, feat_l= input
        nH= self.nhead
        # Global Ave Pooling:channel attention
        b1,c1,_,_= feat_h.shape
        b2,c2,_,_= feat_l.shape

        feat_h_wgh= self.gap(feat_h).view(b1,c1,nH)
        feat_l_wgh = self.gap(feat_l).view(b2,c2,nH)
        feat_wgh= torch.cat([feat_h_wgh,feat_l_wgh],dim=1)   # b c1+c2 nH

        # feat weight multi
        feat_wgh= self.wghLinear(feat_wgh).view(b1,-1)
        feat_wgh= self.wghLinear2(feat_wgh)
        feat_h= feat_h* feat_wgh[:,0:c1].view(b1,c1,1,1)
        feat_l= feat_l* feat_wgh[:,c1:c1+c2].view(b2,c2,1,1)

        # add after
        if self.isup== True:
            feat_h_up= self.upConv(feat_h)
            feat= torch.cat([feat_h_up, feat_l], dim=1)
            feat= self.bottle(feat)
        else:
            feat_l_down= self.downConv(feat_l)
            feat= torch.cat([feat_l_down, feat_h], dim=1)
            feat= self.bottle(feat)
        return feat


class heightAndwidthAttention(nn.Module):

    def __init__(self, ch1, ch2, r_factor=64, mid_hw=48):
        super().__init__()
        self.propaga= nn.Conv2d(in_channels=ch1,out_channels=ch2,kernel_size=1,stride=1)
        # rowpool (b,c,h,w)->(b,c,mid_hw,1)
        self.rowpool = nn.AdaptiveAvgPool2d((mid_hw, 1))

        ch_mid = ch1// r_factor
        self.chReduceRow = nn.Sequential(
            nn.Conv1d(in_channels=ch1, out_channels=ch_mid,kernel_size=1, stride=1, padding=0),
            nn.Hardswish()
        )
        self.chExpandRow = nn.Sequential(
            nn.Conv1d(in_channels=ch_mid, out_channels=ch2, kernel_size=1, stride=1, padding=0),
            nn.Hardswish()
        )

        # colpool (b,c,h,w)->(b,c,1,mid_hw)
        self.colpool = nn.AdaptiveAvgPool2d((1, mid_hw))

        self.chReduceCol = nn.Sequential(
            nn.Conv1d(in_channels=ch1, out_channels=ch_mid, kernel_size=1, stride=1, padding=0),
            nn.Hardswish()
        )
        self.chExpandCol = nn.Sequential(
            nn.Conv1d(in_channels=ch_mid, out_channels=ch2, kernel_size=1, stride=1, padding=0),
            nn.Hardswish()
        )
        # channel ch*2->ch
        self.convout= nn.Conv2d(in_channels=ch2*2, out_channels=ch2, kernel_size=1, stride=1, padding=0)

    def forward(self, featlist):
        """
        feat:b c h w
        height attention and width attention
        """
        feat1, feat2= featlist
        featout= self.propaga(feat1)   # output size
        _, _, H, W= featout.shape
        featHAtt = self.rowpool(feat1).squeeze(3)
        featHAtt = self.chReduceRow(featHAtt)
        featHAtt = self.chExpandRow(featHAtt)
        featHAtt = F.interpolate(featHAtt, size=H, mode='linear')

        featWAtt = self.colpool(feat1).squeeze(2)
        featWAtt = self.chReduceCol(featWAtt)
        featWAtt = self.chExpandCol(featWAtt)
        featWAtt = F.interpolate(featWAtt, size=W, mode='linear')

        out1 = torch.mul(featout, featHAtt.unsqueeze(3))
        out2 = torch.mul(featout, featWAtt.unsqueeze(2))

        featCat= torch.cat([out1+out2, feat2], dim=1)
        featCat= self.convout(featCat)

        return featCat


class GateConv(nn.Module):
    # depthwise conv
    def __init__(self, c1, c2, g):
        super().__init__()
        self.conv= nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, stride=1, groups=g)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    x1= torch.rand(2, 512, 16, 16)
    x2= torch.rand(2, 128, 64, 64)

    model= gateAttention(ch_in=512,ch_out=128)
    print(model([x1,x2]).shape)