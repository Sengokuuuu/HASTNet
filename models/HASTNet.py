import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from sam2.build_sam import build_sam2
from functools import partial
from timm.models.layers import trunc_normal_tf_, trunc_normal_
from timm.models.helpers import named_apply
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from typing import Optional, Callable
from timm.layers import DropPath
import warnings
warnings.filterwarnings('ignore')


class EnhancedShiftModule(nn.Module):
    def __init__(self, shift_size=1):
        super(EnhancedShiftModule, self).__init__()
        self.shift_size = shift_size
        
    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1 = torch.roll(x1, self.shift_size, dims=2)
        x2 = torch.roll(x2, -self.shift_size, dims=2)
        x3 = torch.roll(x3, self.shift_size, dims=3)
        x4 = torch.roll(x4, -self.shift_size, dims=3)
        shifted_x = torch.cat([x1, x2, x3, x4], dim=1)
        return shifted_x


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x)
        spatial_weights = spatial_weights.reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max_val = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max_val), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class STFCM(nn.Module):
    def __init__(self, dim, shift_size=1, reduction=1, eps=1e-8):
        super(STFCM, self).__init__()
        self.eps = eps
        self.shift_module = EnhancedShiftModule(shift_size)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.output_fusion = nn.Conv2d(dim, dim, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        original_feature = x
        shifted_feature = self.shift_module(x)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        spatial_weights = self.spatial_weights(original_feature, shifted_feature)
        original_enhanced = original_feature + fuse_weights[0] * spatial_weights[1] * shifted_feature
        shifted_enhanced = shifted_feature + fuse_weights[0] * spatial_weights[0] * original_feature
        channel_weights = self.channel_weights(original_enhanced, shifted_enhanced)
        final_original = original_enhanced + fuse_weights[1] * channel_weights[1] * shifted_enhanced
        final_shifted = shifted_enhanced + fuse_weights[1] * channel_weights[0] * original_enhanced
        fused_output = (final_original + final_shifted) / 2
        output = self.output_fusion(fused_output)
        return output


class SimpleUpsample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(SimpleUpsample, self).__init__()
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
      
    def forward(self, x):
        return self.upsample(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) 
        pattn1 = pattn1.unsqueeze(dim=2) 
        x2 = torch.cat([x, pattn1], dim=2) 
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CSAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CSAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        B, C, H, W = x.shape
        X = x
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) 
        out_conv = out_conv.squeeze(2)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        gate_output = self.gate(X)
        if torch.isnan(gate_output).any():
            gate_output = torch.nan_to_num(gate_output, nan=0.0) 
        dynamic_k = int(C * gate_output.view(B, -1).mean())
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(B, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
        return output


class MAAM(nn.Module): 
    def __init__(self, dim):
        super(MAAM, self).__init__()
        self.CSAFM = CSAFM(dim)
        self.PixelAttention = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        initial = x + y
        pattn1 = self.CSAFM(initial)
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MRFAConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.v2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.dim = dim
        
    def forward(self, x):
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)
        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)
        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)
        return x


class ConvAtt(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)
        out_att = out_lk + out_dk
        out = torch.cat([out_att, F_idt], dim=1)
        out = self.fusion(out)
        return out


class HRFA(nn.Module):
    def __init__(self, dim, lk_size=[7, 9, 11]):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = ConvAtt(in_channels=dim // 4, lk_size=lk_size[0])
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = ConvAtt(in_channels=dim // 2, lk_size=lk_size[1])
        self.v2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = ConvAtt(in_channels=dim * 3 // 4, lk_size=lk_size[2])
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.dim = dim
        
    def forward(self, x):
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)
        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)
        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)
        return x


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class Decoder(nn.Module):
    def __init__(self, channels=[1152, 576, 288, 144], low_dim=8, fusion_mode='adaptive'):
        super(Decoder, self).__init__()
      
        self.upsample4to3 = SimpleUpsample(in_channels=channels[0], scale=2)
        self.channel_adjust4to3 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        self.upsample3to2 = SimpleUpsample(in_channels=channels[1], scale=2)
        self.channel_adjust3to2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )
        self.upsample2to1 = SimpleUpsample(in_channels=channels[2], scale=2)
        self.channel_adjust2to1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True)
        )

        self.maam3 = MAAM(dim=channels[1])
        self.maam2 = MAAM(dim=channels[2])
        self.maam1 = MAAM(dim=channels[3])

        self.hrfa4 = HRFA(dim=channels[0], lk_size=[7, 9, 11])
        self.hrfa3 = HRFA(dim=channels[1], lk_size=[5, 7, 9])
        self.hrfa2 = HRFA(dim=channels[2], lk_size=[3, 5, 7])
        self.hrfa1 = HRFA(dim=channels[3], lk_size=[3, 5, 7])

        self.stfcm4 = STFCM(dim=channels[0], shift_size=1) 
        self.stfcm3 = STFCM(dim=channels[1], shift_size=1) 
        self.stfcm2 = STFCM(dim=channels[2], shift_size=1) 
        self.stfcm1 = STFCM(dim=channels[3], shift_size=1)
        
    def forward(self, x, skips):
        d4 = self.hrfa4(x)
        d4 = self.stfcm4(d4)

        d3 = self.upsample4to3(d4) 
        d3 = self.channel_adjust4to3(d3) 
        d3 = self.maam3(d3, skips[0])
        d3 = self.hrfa3(d3)
        d3 = self.stfcm3(d3)

        d2 = self.upsample3to2(d3)
        d2 = self.channel_adjust3to2(d2)
        d2 = self.maam2(d2, skips[1])
        d2 = self.hrfa2(d2)
        d2 = self.stfcm2(d2)

        d1 = self.upsample2to1(d2)
        d1 = self.channel_adjust2to1(d1)
        d1 = self.maam1(d1, skips[2])
        d1 = self.hrfa1(d1)
        d1 = self.stfcm1(d1)
        return [d4, d3, d2, d1]


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 8),
            nn.GELU(),
            nn.Linear(8, dim),
            nn.GELU()
        )
        
    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class HASTNet(nn.Module):
    def __init__(self, num_cls=4, low_dim=8, fusion_mode='adaptive'):
        super(HASTNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        checkpoint_path = "sam2_hiera_large.pt"
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}...")
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)
        channels = [1152, 576, 288, 144]
        self.decoder = Decoder(channels=channels, low_dim=low_dim, fusion_mode=fusion_mode)
        self.out_head4 = nn.Conv2d(channels[0], num_cls, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_cls, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_cls, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_cls, 1)
        
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        dec_outs = self.decoder(x4, [x3, x2, x1])
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])
        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=2, mode='bilinear')
        return [p4, p3, p2, p1]


def get_module_params(module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params
    

def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


if __name__ == "__main__":
    print("=" * 50)
    try:
        simple_upsample = SimpleUpsample(in_channels=32, scale=2)
        decoder = Decoder(channels=[1152, 576, 288, 144], fusion_mode='adaptive')
        model = HASTNet(num_cls=4, low_dim=8)
        print("\nFull Model Analysis:")
        print_model_params(model)
    except Exception as e:
        decoder = Decoder(channels=[1152, 576, 288, 144], fusion_mode='adaptive')