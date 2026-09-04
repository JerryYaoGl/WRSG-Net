import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from einops import rearrange



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # 沿宽方向池化 -> [B, C, H, 1]
        x_h = x.mean(dim=3, keepdim=True)

        # 沿高方向池化 -> [B, C, 1, W]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out



class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(8, channels // ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        attn = torch.sigmoid(avg_out + max_out)
        return attn



class ScaleModule(nn.Module):
    def __init__(self, shape, init=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(shape) * init)

    def forward(self, x):
        return x * self.scale


class ParametricDWT(nn.Module):
    def __init__(self, in_channels, wt_type='db1'):
        super().__init__()

        wavelet = pywt.Wavelet(wt_type)
        lo = torch.tensor(wavelet.dec_lo, dtype=torch.float32)
        hi = torch.tensor(wavelet.dec_hi, dtype=torch.float32)

        # 2D 小波核：LL / LH / HL / HH
        ll = torch.ger(lo, lo)
        lh = torch.ger(lo, hi)
        hl = torch.ger(hi, lo)
        hh = torch.ger(hi, hi)

        k = ll.shape[0]

        self.k = k
        self.in_channels = in_channels

        self.w_ll = nn.Parameter(ll.view(1, 1, k, k).repeat(in_channels, 1, 1, 1))
        self.w_lh = nn.Parameter(lh.view(1, 1, k, k).repeat(in_channels, 1, 1, 1))
        self.w_hl = nn.Parameter(hl.view(1, 1, k, k).repeat(in_channels, 1, 1, 1))
        self.w_hh = nn.Parameter(hh.view(1, 1, k, k).repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape

        # 保证偶数尺寸
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        pad = self.k // 2 - 1 if self.k > 2 else 0

        ll = F.conv2d(x, self.w_ll, stride=2, padding=pad, groups=self.in_channels)
        lh = F.conv2d(x, self.w_lh, stride=2, padding=pad, groups=self.in_channels)
        hl = F.conv2d(x, self.w_hl, stride=2, padding=pad, groups=self.in_channels)
        hh = F.conv2d(x, self.w_hh, stride=2, padding=pad, groups=self.in_channels)

        return ll, lh, hl, hh

# UNet 基础模块
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

#Parametric Wavelet Refined Encoder
class PWREC(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type='db1'):
        super().__init__()

        # -------- x_down：图中 3×3 stride=2 --------
        self.x_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        # -------- 参数化小波 --------
        self.pdwt = ParametricDWT(in_channels, wt_type=wt_type)

        self.wavelet_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        # -------- 融合后通道注意力 --------
        self.channel_att = ChannelAttention(out_channels)

        # -------- f_conv分支--------
        self.f_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.scale = ScaleModule([1, out_channels, 1, 1], init=0.1)
        self.coord_att = CoordAttention(out_channels)
        
        self.enc = conv_block(out_channels, out_channels)

    def forward(self, x):
        # 1) x_down分支
        x_down = self.x_down(x)

        # 2) 小波分支
        ll, lh, hl, hh = self.pdwt(x)
        f_cat = torch.cat([ll, lh, hl, hh], dim=1)
        f_cat = self.wavelet_fuse(f_cat)

        f_wave = x_down + f_cat

        # 3) 通道注意力
        f_ca = self.channel_att(f_wave)

        # 4) f_conv 分支
        f_conv = self.f_conv(x)

        # 5) Learnable Scale
        f_s = self.scale(f_wave)

        # 6) 逐元素调制
        out = (f_conv * f_ca) * f_s

        # 8) CoordAttention
        out = self.coord_att(out)

        # 9) 卷积块
        out = self.enc(out)
        
        return out


#Inter-scale-Scale Information propagation Module
class ISIPM(nn.Module):
    def __init__(self, cur_dim, prev_dim=None, next_dim=None):
        super().__init__()

        self.cur_dim = cur_dim
        self.prev_dim = prev_dim
        self.next_dim = next_dim

        # 上一层特征先通道对齐
        if prev_dim is not None:
            self.prev_align = nn.Conv2d(prev_dim, cur_dim, kernel_size=1, bias=False)
        else:
            self.prev_align = None

        # X = concat(E_i, M_{i-1})
        self.conv_u0 = nn.Conv2d(cur_dim * 2, cur_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm(cur_dim)
        self.conv_u2 = nn.Conv2d(cur_dim, cur_dim, kernel_size=1, bias=False)

        # 两条初始多尺度支路
        self.top_in = nn.Conv2d(cur_dim, cur_dim // 2, kernel_size=1, bias=False)
        self.bot_in = nn.Conv2d(cur_dim, cur_dim // 2, kernel_size=1, bias=False)

        self.dw5_a = nn.Conv2d(cur_dim // 2, cur_dim // 2, kernel_size=5, stride=1, padding=2,
                               groups=cur_dim // 2, bias=False)
        self.dw3_a = nn.Conv2d(cur_dim // 2, cur_dim // 2, kernel_size=3, stride=1, padding=1,
                               groups=cur_dim // 2, bias=False)

        # 第二次多尺度交互
        self.dw5_b = nn.Conv2d(cur_dim, cur_dim, kernel_size=5, stride=1, padding=2,
                               groups=cur_dim, bias=False)
        self.dw3_b = nn.Conv2d(cur_dim, cur_dim, kernel_size=3, stride=1, padding=1,
                               groups=cur_dim, bias=False)

        # 输出投影 G
        self.proj_g = nn.Conv2d(cur_dim * 2, cur_dim, kernel_size=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(cur_dim)

        # 右侧Conv 3×3 / s2 -> Y
        if next_dim is not None:
            self.down = nn.Sequential(
                nn.Conv2d(cur_dim, next_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.down = None

    def forward(self, e_i, m_prev=None):
        b, c, h, w = e_i.shape

        # -------- 构造 X = [E_i, M_{i-1}] --------
        if m_prev is None:
            m_prev = torch.zeros_like(e_i)
        else:
            if m_prev.shape[2:] != e_i.shape[2:]:
                m_prev = F.interpolate(m_prev, size=e_i.shape[2:], mode='bilinear', align_corners=False)  #不会执行
            m_prev = self.prev_align(m_prev)

        x = torch.cat([e_i, m_prev], dim=1)

        # -------- U0 / U1 / U2 --------
        u0 = self.conv_u0(x)  #通道对齐
        u1 = self.norm(u0)
        u2 = self.conv_u2(u1)

        # -------- 第一轮多尺度分支 --------
        top = self.act(self.dw5_a(self.top_in(u2)))
        bot = self.act(self.dw3_a(self.bot_in(u2)))

        # -------- 构造 F1 / F2--------
        f1 = torch.cat([top, bot], dim=1)
        f2 = torch.cat([bot, top], dim=1)

        # -------- 第二轮多尺度卷积 --------
        f1 = self.act(self.dw5_b(f1))
        f2 = self.act(self.dw3_b(f2))

        # -------- 拼接 -> Conv1×1 -> G --------
        g = self.proj_g(torch.cat([f1, f2], dim=1))
        g = self.act(self.bn(g))

        # -------- 残差得到 Z --------
        z = g + u1

        # -------- 向下传递得到 Y --------
        if self.down is not None:
            y = self.down(z)
        else:
            y = None

        return z, y



class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

#Semantic Guided Decoder
class SGDC(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.Up = up_conv(F_g, F_l)
                
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.Up_conv = conv_block(F_g, F_l)       
       
    def forward(self, g, x):
        
        g_up = self.Up(g)
        
        #Decoder samantic guidance
        psi = self.psi(self.relu(self.W_g(g_up) + self.W_x(x)))       
        att = x * psi
        con = torch.cat([att, g_up], dim=1) 
        g_con = self.Up_conv(con)
          
        return  g_con

class WRSG_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super(WRSG_Net, self).__init__()

        # ---------------- Parametric Wavelet Refined Encoder ----------------
        self.encoder1 = conv_block(img_ch, 64)
        self.encoder2 = PWREC(64, 128)
        self.encoder3 = PWREC(128, 256)
        self.encoder4 = PWREC(256, 512)
        self.encoder5 = PWREC(512, 1024)

        # ---------------- Inter-scale Information propagation module ----------------
        self.isipm1 = ISIPM(cur_dim=64,  prev_dim=None, next_dim=128)
        self.isipm2 = ISIPM(cur_dim=128, prev_dim=128,  next_dim=256)
        self.isipm3 = ISIPM(cur_dim=256, prev_dim=256,  next_dim=512)
        self.isipm4 = ISIPM(cur_dim=512, prev_dim=512,  next_dim=None)

        # ---------------- Semantic Guided Decoder ----------------
        self.decoder5 = SGDC(F_g=1024, F_l=512, F_int=256)
        self.decoder4 = SGDC(F_g=512, F_l=256, F_int=128)
        self.decoder3 = SGDC(F_g=256, F_l=128, F_int=64)
        self.decoder2 = SGDC(F_g=128, F_l=64, F_int=32)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
 
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        #Inter-scale Information propagation 
        z1, y1 = self.isipm1(e1, None)
        z2, y2 = self.isipm2(e2, y1)
        z3, y3 = self.isipm3(e3, y2)
        z4, _  = self.isipm4(e4, y3)

        # Decoder
        d5 = self.decoder5(e5, z4)
        d4 = self.decoder4(d5, z3)
        d3 = self.decoder3(d4, z2)
        d2 = self.decoder2(d3, z1)

        out = self.Conv_1x1(d2)
        
        return F.softmax(out, dim=1)