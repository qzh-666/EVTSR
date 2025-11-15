import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from basicsr.utils.registry import ARCH_REGISTRY
from .unet_arch import UNet
from .arch_util import flow_warp, ConvResidualBlocks, SmallUpdateBlock
from .spynet_arch import SpyNet
import matplotlib.pyplot as plt
import numpy as np
from basicsr.textprior.Position_aware_module import PositionAwareModule,Location_enhancement_Multimodal_alignment
SHUT_BN = False
@ARCH_REGISTRY.register()

class EVTSR(nn.Module):
    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.block_expansion = 2
        self.imgconv= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.position_prior = PositionAwareModule()
        self.guidanceGen = Location_enhancement_Multimodal_alignment()

        self.DMFF = DualModalFeatureFusion(input_channels=64, reduction_ratio=16)
        self.TEB = dynamic_filter_channel(inchannels=64)
        self.EGH = EGHM()
        self.TGL = FeatureWiseAffine(in_channels=64, out_channels=64, pool_size=(4, 4))

        self.ERDMF = Event_RGB_Dynamic_Fusion_Module(64,64)
        self.outconv= nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    def visualize_input(self, imgs):
        img_data = imgs[0, 0].cpu().detach().numpy()
        img_data = np.transpose(img_data, (1, 2, 0))
        plt.imshow(img_data)
        plt.axis('off')
        plt.savefig('img_result/input_rgb_image.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def visualize_output(self, imgs):
        img_data = imgs[0, 0].cpu().detach().numpy()
        img_data = np.transpose(img_data, (1, 2, 0))
        plt.imshow(img_data)
        plt.axis('off')
        plt.savefig('img_result/input_rgb_image.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def visualize_output(self, imgs, index=0):
        img_data = imgs[index].cpu().detach().numpy()
        img_data = np.transpose(img_data, (1, 2, 0))
        plt.imshow(img_data)
        plt.axis('off')
        plt.savefig(f'img_result/input_rgb_output.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def forward(self, imgs, voxels_f, voxels_b):
        self.visualize_input(imgs)
        b, n, c, h, w = imgs.size()
        x1=imgs.squeeze(1)
        img_feature = self.imgconv(x1)

        low,high = self.TEB(img_feature)

        if img_feature.size(2) != 32 or img_feature.size(3) != 32:
            img_feature1 = F.interpolate(img_feature, size=(32, 32), mode='bicubic', align_corners=False)
        else:
            img_feature1=img_feature

        voxels = torch.cat((voxels_f, voxels_b), dim=1)
        b, n, c, h, w = voxels.size()
        conv1x1 = nn.Conv2d(in_channels=n*c, out_channels=64, kernel_size=1).to(voxels.device)
        event_guide = conv1x1(voxels.view(b, n*c, w, h))

        high_pro = self.EGH(high,event_guide)

        x_resized = F.interpolate(x1, size=(64, 64), mode='bicubic', align_corners=False)

        pos_prior = self.position_prior(x_resized)
        text_guide, pr_weights = self.guidanceGen(pos_prior, img_feature1)

        text_guide= F.interpolate(text_guide, size=(h, w), mode='bicubic', align_corners=False)
        low_pro = self.TGL(low, text_guide)

        img_pro = high_pro + low_pro

        img_promax = self.ERDMF(img_pro,event_guide)

        output = self.outconv(img_promax)

        self.visualize_output(output)

        return output.unsqueeze(1)+imgs

class Event_RGB_Dynamic_Fusion_Module(nn.Module):
    def __init__(self, rgb_channels, event_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.kernel_gen = nn.Sequential(
            nn.Conv2d(rgb_channels + event_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, rgb_channels * kernel_size * kernel_size, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, rgb_feature, event_feature):
        assert rgb_feature.shape[2:] == event_feature.shape[2:], "RGB 和 Event 特征的空间维度必须一致"

        fused_features = torch.cat([rgb_feature, event_feature], dim=1)

        B, C, H, W = rgb_feature.shape
        dynamic_kernels = self.kernel_gen(fused_features)
        dynamic_kernels = dynamic_kernels.view(B, C, self.kernel_size, self.kernel_size, H, W)

        rgb_padded = F.pad(rgb_feature, (self.padding, self.    padding, self.padding, self.padding), mode='reflect')
        patches = rgb_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = patches.permute(0, 1, 4, 5, 2, 3)

        output = torch.einsum('bckkhw,bckkhw->bchw', patches, dynamic_kernels)

        return output

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels,  pool_size=(4, 4)):
        super().__init__()
        self.pool_size = pool_size

        pooled_channels = 64 * pool_size[0] * pool_size[1]

        self.MLP = nn.Sequential(
            nn.Linear(pooled_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * 2)
        )
        self.attn = TIAM(dim=64, head_num=16)
    def forward(self, x, text_embed):
        x = self.attn(x,text_embed)
        batch = x.shape[0]

        text_embed = F.adaptive_avg_pool2d(text_embed, self.pool_size)

        text_embed = text_embed.view(batch, -1)

        gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
        x = (1 + gamma) * x + beta
        return x

class TIAM(nn.Module):
    def __init__(self, dim=64, head_num=16, chunk_size=128):
        super(TIAM, self).__init__()

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scale = math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.act_fn = nn.GELU()

        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

        self.chunk_size = chunk_size

    def chunked_attention(self, Q, K, V, chunk_size):
        B, head_num, N, head_dim = Q.size()
        attention_output = torch.zeros_like(V)

        for i in range(0, N, chunk_size):
            start = i
            end = min(i + chunk_size, N)

            Q_chunk = Q[:, :, start:end, :]
            K_chunk = K[:, :, start:end, :]
            V_chunk = V[:, :, start:end, :]

            attention_scores = torch.matmul(Q_chunk, K_chunk.transpose(-1, -2)) / self.scale
            attention_probs = self.softmax(attention_scores)

            chunk_output = torch.matmul(attention_probs, V_chunk)

            attention_output[:, :, start:end, :] = chunk_output

        return attention_output

    def forward(self, x, prior):
        B, C, H, W = x.size()
        original_x = x

        N = H * W
        prior_flat = prior.view(B, C, N).permute(0, 2, 1)
        x_flat = x.view(B, C, N).permute(0, 2, 1)

        prior_norm = self.norm1(prior_flat)

        Q = self.query_proj(prior_norm)
        K = self.key_proj(x_flat)
        V = self.value_proj(x_flat)

        Q = Q.view(B, N, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.head_num, self.head_dim).transpose(1, 2)

        attention_output = self.chunked_attention(Q, K, V, self.chunk_size)

        attention_output = attention_output.transpose(1, 2).contiguous().view(B, N, self.dim)
        attention_output = self.out_proj(attention_output)

        x = attention_output + prior_flat

        x_norm = self.norm2(x)

        x_ffn = self.fc2(self.act_fn(self.fc1(x_norm)))
        x = x_ffn + x

        x = x.permute(0, 2, 1).contiguous().view(B, self.dim, H, W)

        x = self.conv_out(x)
        output = x + original_x

        return output

class EGHM(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=128, filter_size=5):
        super().__init__()

        self.filter_size = filter_size

        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)

        self.norm = nn.LayerNorm(hidden_channels)

        self.dynamic_filter = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * (filter_size ** 2), kernel_size=1, bias=False)
        )

        self.conv_spatial = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

        self.conv_output = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)

    def apply_dynamic_filter(self, features, dynamic_filter):
        B, C, H, W = features.size()
        K = self.filter_size

        dynamic_filter = dynamic_filter.view(B, C, K * K, H, W)

        dynamic_filter = dynamic_filter.view(B, C, K * K, H * W)

        features_unfold = F.unfold(features, kernel_size=K, padding=K // 2)

        features_unfold = features_unfold.view(B, C, K * K, H * W)

        filtered = (features_unfold * dynamic_filter).sum(dim=2)

        filtered = filtered.view(B, C, H, W)

        return filtered

    def forward(self, rgb_high, event):
        batch_size, _, height, width = rgb_high.size()

        rgb_feat = self.conv(rgb_high)
        event_feat = self.conv(event)

        rgb_feat = self.norm(rgb_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        event_feat = self.norm(event_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        rgb_filter = self.dynamic_filter(rgb_feat)
        event_filter = self.dynamic_filter(event_feat)

        rgb_enhanced = self.apply_dynamic_filter(rgb_feat, rgb_filter)
        event_enhanced = self.apply_dynamic_filter(event_feat, event_filter)

        rgb_filtered = rgb_enhanced * torch.sigmoid(self.conv_spatial(event_enhanced))

        rgb_fused = rgb_enhanced + rgb_filtered + rgb_feat

        rgb_norm = self.norm(rgb_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        rgb_output = self.conv_output(rgb_norm)

        return rgb_output

class dynamic_filter_channel(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group*kernel_size**2, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.act_gate  = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.ln = nn.LayerNorm(group * kernel_size**2)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter1 = self.ap_1(x)
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        n, c, h, w = low_filter.shape
        low_filter = low_filter.view(n, c)
        low_filter = self.ln(low_filter)
        low_filter = low_filter.view(n, c, 1, 1)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high
