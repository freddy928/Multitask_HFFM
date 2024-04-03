import torch
import math
from torch import nn
from torch.nn import functional as F
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Conv_decoder


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class HierarchicalFFM(nn.Module):
    """
    Hierarchical feature fusion module for stage x3 and stage x4
    """
    def __init__(self, ch_x3, ch_x4, nhead=3, head_dim=64, patch_size=2, attn_drop=0., proj_drop=0.):
        super().__init__()

        # 1. multi-head cross attention
        head_dim= ch_x3
        ch_inner= nhead* head_dim
        self.nhead= nhead
        self.head_dim= head_dim
        self.scale = head_dim ** -0.5

        self.up= nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.x4_upconv = BottleneckCSP(ch_x4, ch_x3, 1, False)

        # patch embedding
        self.patch_size= patch_size
        self.patch= PatchEmbed(patch_size=patch_size, in_chans=ch_x3, embed_dim=ch_x3)   # reduce compute cost

        self.v_linear= nn.Linear(ch_x3, ch_inner)
        self.qk_linear= nn.Linear(ch_x3, 2*ch_inner)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(ch_inner, ch_inner)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear_restore= nn.Linear(ch_inner, ch_x3)
        self.patch_up= nn.Upsample(size=None, scale_factor=patch_size, mode='nearest')

        # 2. gap channel attention
        self.cat_conv= BottleneckCSP(ch_x3*2, ch_x3, 1, False)
        self.gap= nn.AdaptiveAvgPool2d(1)

        self.out= BottleneckCSP(ch_x3, ch_x3, 3, False)

    def forward(self, x):
        '''
        x3: b c h/8 w/8
        x4: b 2c h/16 w/16
        '''
        '''1. cross attention'''
        x3, x4= x
        x4_up= self.x4_upconv(self.up(x4))
        x3_add= x3+ x4_up

        b, ch3, h, w= x3.shape
        v_x3= self.patch(x3)
        x3_add = self.patch(x3_add)

        v= self.v_linear(v_x3).view(b, h*w//self.patch_size**2, self.nhead, self.head_dim).permute(0, 2, 1, 3)   # b nh n c
        qk= self.qk_linear(x3_add).view(b, -1, self.nhead, self.head_dim, 2).permute(4, 0, 2, 1, 3)    # 2 b n c ->unbind-> b n c
        q,k= qk.unbind(0)

        # qkv attention b nh n c
        attn= (q @ k.transpose(-2, -1)) * self.scale
        attn= attn.softmax(dim=-1)
        attn= self.attn_drop(attn)
        x_mt= (attn @ v).transpose(1, 2).reshape(b, -1, self.nhead* self.head_dim)

        x_mt= self.linear_restore(x_mt).reshape(b, h//self.patch_size, w// self.patch_size, ch3).permute(0, 3, 1, 2)
        if self.patch_size> 1:
            x_mt= self.patch_up(x_mt)
        x3_add_out= x3+ x_mt

        '''2. gap channel attention'''
        x3_cat= torch.cat([x3_add_out, x4_up], dim=1)
        x3_cat= self.cat_conv(x3_cat)
        x4_gp= self.gap(x4_up)
        x3_cat_out= x3_cat* x4_gp

        out= x3_add_out+ x3_cat_out
        out= self.out(out)+ out

        return out


class CrossLayerAttention(nn.Module):

    def __init__(self, ch_low, ch_upp, is_upper= False, is_up=True, nhead=3, head_dim=64, pooling='mean', attn_drop=0., prob_drop=0.):
        super().__init__()

        head_dim = ch_low
        ch_inner = nhead * head_dim
        # is_up=True: output stage3; is_up=False: output stage5
        self.is_up= is_up
        if self.is_up== False:
            ch_temp= ch_low
            ch_low= ch_upp
            ch_upp= ch_temp

        self.xu_linearfor= nn.Linear(ch_upp,ch_inner)
        self.mha= nn.MultiheadAttention(embed_dim=head_dim*nhead, num_heads=nhead)
        self.xu_linearback= nn.Linear(ch_inner, ch_upp)

        self.nhead= nhead
        self.head_dim= head_dim
        self.scale = head_dim ** -0.5

        self.pooling = pooling
        self.prob_drop = prob_drop

        self.H_initial = 96
        if self.pooling == 'mean':
            # print("##### average pooling")
            self.rowpool = nn.AdaptiveAvgPool2d((self.H_initial, 1))
        else:
            # print("##### max pooling")
            self.rowpool = nn.AdaptiveMaxPool2d((self.H_initial, 1))

        self.attn_drop = nn.Dropout(attn_drop)
        self.dropout = nn.Dropout2d(self.prob_drop)

        # cross attention
        self.ch_l_reduceconv_k= nn.Sequential(
            nn.Conv1d(in_channels=ch_low, out_channels=ch_inner,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(ch_inner),
            nn.Hardswish(inplace=True))

        self.is_upper= is_upper
        if is_upper== False:
            self.ch_reduceconv_v= nn.Sequential(
                nn.Conv1d(in_channels=ch_low, out_channels=ch_inner,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(ch_inner),
                nn.Hardswish(inplace=True))

            self.ch_reduceconv_q = nn.Sequential(
                nn.Conv1d(in_channels=ch_upp, out_channels=ch_inner,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(ch_inner),
                nn.Hardswish(inplace=True)
            )
        else:
            self.ch_reduceconv_v= nn.Sequential(
                nn.Conv1d(in_channels=ch_upp, out_channels=ch_inner,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(ch_inner),
                nn.Hardswish(inplace=True))
            self.ch_reduceconv_q = nn.Sequential(
                nn.Conv1d(in_channels=ch_low, out_channels=ch_inner,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(ch_inner),
                nn.Hardswish(inplace=True)
            )

        self.restoreconv= nn.Sequential(
            nn.Conv1d(in_channels=ch_inner, out_channels=ch_low,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(ch_low),
            nn.Hardswish(inplace=True))

        self.catconv= BottleneckCSP(ch_low*2, ch_low, 3, False)
        self.out= BottleneckCSP(ch_low, ch_low, 3, False)

    def forward(self, x):
        '''
        x_l: b c h/8 w/8
        x_u: b 4c h/32 w/32
        '''
        if self.is_up== True:
            x_l, x_u= x
        else:
            x_u, x_l= x

        '''1.upper layer multi-head attention'''
        b, cu, hu, wu= x_u.shape
        xu_linear= x_u.view(b, cu, hu*wu).permute(0, 2, 1)
        xu_linear= self.xu_linearfor(xu_linear)
        xu_out= self.mha(xu_linear,xu_linear,xu_linear)[0]
        x_u= self.xu_linearback(xu_out).view(b, hu, wu, cu).permute(0, 3, 1, 2)

        _, cl, h, w= x_l.shape
        x_l_pool= self.rowpool(x_l).squeeze(3)   # b cl 48
        x_u_pool= self.rowpool(x_u).squeeze(3)   # b cu 48

        x_l_pool_k= self.ch_l_reduceconv_k(x_l_pool)

        if self.is_upper== False:
            x_pool_v = self.ch_reduceconv_v(x_l_pool)
            x_pool_q = self.ch_reduceconv_q(x_u_pool)
        else:
            x_pool_v = self.ch_reduceconv_v(x_u_pool)
            x_pool_q = self.ch_reduceconv_q(x_l_pool)

        '''2.cross layer attention'''

        q= x_pool_q.view(b, self.nhead, self.head_dim, self.H_initial).permute(0, 1, 3, 2)
        k= x_l_pool_k.view(b, self.nhead, self.head_dim, self.H_initial).permute(0, 1, 3, 2)
        v= x_pool_v.view(b, self.nhead, self.head_dim, self.H_initial).permute(0, 1, 3, 2)

        # qkv attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn2 = self.attn_drop(attn)
        x_mt = (attn @ v).transpose(1, 2).reshape(b, self.H_initial, self.nhead* self.head_dim)
        x_mt = self.restoreconv(x_mt.permute(0, 2, 1))
        xd = F.interpolate(x_mt, size=h, mode='linear')

        out = torch.mul(x_l, xd.unsqueeze(3))

        out= self.out(out)+ out
        return out


if __name__ == '__main__':
    x3= torch.rand([2,128,32,32])
    x4= torch.rand([2,256,16,16])
    x5= torch.rand([2,512,8,8])
    model= HierarchicalFFM(ch_x3=128,ch_x4=256)
    xl= model([x3,x4])
    #print(xl.shape)

    model2= CrossLayerAttention(128,512,False)
    print(model2([xl,x5]).shape)