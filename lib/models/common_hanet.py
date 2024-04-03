import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def initialize_embedding(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.zero_() #original


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        if d_hid > 50:
            cycle = 10
        elif d_hid > 5:
            cycle = 100
        else:
            cycle = 10000
        cycle = 10 if d_hid > 50 else 100
        return position / np.power(cycle, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class PosEmbedding2D(nn.Module):

    def __init__(self, pos_rfactor, dim):
        super(PosEmbedding2D, self).__init__()

        self.pos_layer_h = nn.Embedding((128 // pos_rfactor) + 1, dim)
        self.pos_layer_w = nn.Embedding((128 // pos_rfactor) + 1, dim)
        initialize_embedding(self.pos_layer_h)
        initialize_embedding(self.pos_layer_w)

    def forward(self, x, pos):
        pos_h, pos_w = pos
        pos_h = pos_h.unsqueeze(1)
        pos_w = pos_w.unsqueeze(1)
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2:], mode='nearest').long()  # B X 1 X H X W
        pos_w = nn.functional.interpolate(pos_w.float(), size=x.shape[2:], mode='nearest').long()  # B X 1 X H X W
        pos_h = self.pos_layer_h(pos_h).transpose(1, 4).squeeze(4)  # B X 1 X H X W X C
        pos_w = self.pos_layer_w(pos_w).transpose(1, 4).squeeze(4)  # B X 1 X H X W X C
        x = x + pos_h + pos_w
        return x


class PosEncoding1D(nn.Module):

    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D, self).__init__()
        # print("use PosEncoding1D")
        self.sel_index = torch.tensor([0])
        pos_enc = (get_sinusoid_encoding_table((128 // pos_rfactor) + 1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor  # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0  # torch.tensor([0]).cuda()
            self.max = 128 // pos_rfactor  # torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos  # B X H X W
        pos_h = pos_h // self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3)  # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()  # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            # pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3) // 1).long(),
                                        min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)
            # pos_h = torch.where(pos_h < self.min_tensor, self.min_tensor, pos_h)
            # pos_h = torch.where(pos_h > self.max_tensor, self.max_tensor, pos_h)

        pos_h = self.pos_layer(pos_h).transpose(1, 3).squeeze(3)  # B X 1 X 48 X 80 > B X 80 X 48 X 1
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight  # 33 X 80
        return x


class PosEncoding1D_2(nn.Module):

    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D_2, self).__init__()
        # print("use PosEncoding1D 2")
        pos_enc = (get_sinusoid_encoding_table((384 // pos_rfactor) + 1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor  # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0  # torch.tensor([0]).cuda()
            self.max = 128 // pos_rfactor  # torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        # x: B C 48
        B, C, H = x.shape
        # pos_h = x.index_select(1, torch.tensor([0]).to(x.device))
        pos_h = torch.arange(0, H).unsqueeze(0).unsqueeze(0).to(x.device)

        if self.training is True and self.pos_noise > 0.0:
            # pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3) // 1).long(),
                                        min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)
            # pos_h = torch.where(pos_h < self.min_tensor, self.min_tensor, pos_h)
            # pos_h = torch.where(pos_h > self.max_tensor, self.max_tensor, pos_h)

        pos_h = pos_h.long()
        a = self.pos_layer(pos_h)
        pos_h = self.pos_layer(pos_h).transpose(1, 3).squeeze(3)  # B X 1 X 48 X 80 > B X 80 X 48 X 1
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight  # 33 X 80
        return x


class PosEmbedding1D(nn.Module):

    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEmbedding1D, self).__init__()
        print("use PosEmbedding1D")
        self.sel_index = torch.tensor([0])
        self.pos_layer = nn.Embedding((128 // pos_rfactor) + 1, dim)
        initialize_embedding(self.pos_layer)
        self.pos_noise = pos_noise
        self.pos_rfactor = pos_rfactor
        self.noise_clamp = 16 // pos_rfactor  # 4: 4, 8: 2, 16: 1

        if pos_noise > 0.0:
            self.min = 0.0  # torch.tensor([0]).cuda()
            self.max = 128 // pos_rfactor  # torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos  # B X H X W
        pos_h = pos_h // self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3)  # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()  # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            # pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3) // 1).long(),
                                        min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)

        pos_h = self.pos_layer(pos_h).transpose(1, 3).squeeze(3)  # B X 1 X 48 X 80 > B X 80 X 48 X 1
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight  # 33 X 80
        return x


class HANet_Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super(HANet_Conv, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(mid_1_channel),
            nn.Hardswish(inplace=True))

        if layer == 2:
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                          kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm1d(mid_2_channel),
                nn.Hardswish(inplace=True))
            self.attention_third = nn.Sequential(
                nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))

        if self.pooling == 'mean':
            # print("##### average pooling")
            self.rowpool = nn.AdaptiveAvgPool2d((128 // pos_rfactor, 1))
        else:
            # print("##### max pooling")
            self.rowpool = nn.AdaptiveMaxPool2d((128 // pos_rfactor, 1))

        if pos_rfactor > 0:
            if is_encoding == 0:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEmbedding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEmbedding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
            elif is_encoding == 1:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
            else:
                print("Not supported position encoding")
                exit()

    def forward(self, x, out, pos=None, return_attention=False, return_posmap=False, attention_loss=False):
        """
            inputs :
                x : input feature maps( B C W H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        H = out.size(2)
        x1d = self.rowpool(x).squeeze(3)

        if pos is not None and self.pos_injection == 1:
            if return_posmap:
                x1d, pos_map1 = self.pos_emb1d_1st(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_1st(x1d, pos)

        if self.dropout_prob > 0:
            x1d = self.dropout(x1d)
        x1d = self.attention_first(x1d)

        if pos is not None and self.pos_injection == 2:
            if return_posmap:
                x1d, pos_map2 = self.pos_emb1d_2nd(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_2nd(x1d, pos)

        x1d = self.attention_second(x1d)

        if self.layer == 3:
            x1d = self.attention_third(x1d)
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)
        else:
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)

        x1d = F.interpolate(x1d, size=H, mode='linear')
        out = torch.mul(out, x1d.unsqueeze(3))

        if return_attention:
            if return_posmap:
                if self.pos_injection == 1:
                    pos_map = (pos_map1)
                elif self.pos_injection == 2:
                    pos_map = (pos_map2)
                return out, x1d, pos_map
            else:
                return out, x1d
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out


class HCAModule(nn.Module):

    def __init__(self, ch_low, ch_upp, kernel_size=3, r_factor=8, layer=3, is_encoding=1, num_head=4,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super().__init__()

        self.pooling = pooling
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.num_head = num_head
        self.sigmoid = nn.Sigmoid()

        self.H_initial = 48
        if self.pooling == 'mean':
            # print("##### average pooling")
            self.rowpool = nn.AdaptiveAvgPool2d((self.H_initial, 1))
        else:
            # print("##### max pooling")
            self.rowpool = nn.AdaptiveMaxPool2d((self.H_initial, 1))

        if r_factor > 0:
            mid_1_channel = math.ceil(ch_upp / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = ch_upp * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
            nn.Conv1d(in_channels=ch_upp, out_channels=mid_1_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(mid_1_channel),
            nn.Hardswish(inplace=True))

        self.H_mid = 8* self.num_head   # multi head

        if layer == 3:
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=self.H_initial, out_channels=self.H_mid,
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(self.H_mid),
                nn.Hardswish(inplace=True))
            self.attention_third1 = nn.Sequential(
                nn.Conv2d(in_channels=self.H_mid// self.num_head, out_channels=self.H_initial// self.num_head,
                          kernel_size=1, stride=1, padding=0, bias=True))
            self.attention_third2 = nn.Sequential(
                nn.Conv2d(in_channels=mid_1_channel, out_channels=ch_low,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))
        self.H_conv = nn.Conv2d(in_channels= mid_1_channel, out_channels= mid_1_channel,
                                kernel_size=3, stride=1, padding=1, bias=True)
        if pos_rfactor > 0:
            if is_encoding == 1:
                self.pos_emb1d = PosEncoding1D_2(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
            else:
                print("Not supported position encoding")
                exit()

    def forward(self, x):
        """
        input:
            x: input feature maps( B C W H)
        returns:
            out: self attention value + input feature
        """
        x_l, x_u= x
        '''swap'''
        x_temp= x_l
        x_l= x_u
        x_u= x_temp

        H = x_u.size(2)
        x1d = self.rowpool(x_l).squeeze(3)     # B C 48
        x1d = self.attention_first(x1d)      # B C_mid 48

        pos = 1
        if pos is not None:
            x1d = self.pos_emb1d(x1d, pos)   # add pos encoding

        if self.dropout_prob > 0:
            x1d = self.dropout(x1d)

        x1d = self.attention_second(x1d.permute(0, 2, 1))     # B H^*nH C_mid
        B, H_x1d, C_mid = x1d.shape
        x1d = x1d.view(B, self.num_head, H_x1d// self.num_head, C_mid).permute(0, 3, 1, 2)   # B C_mid nH H
        x1d = self.H_conv(x1d)+ x1d      # skip

        if self.layer == 3:
            x1d = self.attention_third1(x1d.permute(0, 3, 1, 2))
            x1d = self.attention_third2(x1d.permute(0, 2, 1, 3))
            x1d = x1d.reshape(B, -1, self.H_initial)
            x1d = self.sigmoid(x1d)

        x1d = F.interpolate(x1d, size=H, mode='linear')
        out = torch.mul(x_u, x1d.unsqueeze(3))

        return out


if __name__ == '__main__':
    x_l= torch.rand([2,128,32,32])
    x_u= torch.rand([2,512,8,8])
    model2= HCAModule(128,512)
    print(model2([x_l,x_u]).shape)