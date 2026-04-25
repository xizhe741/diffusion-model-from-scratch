import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.model.modules import (
    sinusoidal_embedding,
    ResBlock,
    self_attention,
    Downsample,
    Upsample,
)




class U_Net(nn.Module):
    def __init__(self, base_channels,embedded_dim, **kwargs):
        super().__init__()

        self.entry = nn.Conv2d(3,base_channels,kernel_size=3, stride=1,padding =1)
        self.exit = nn.Conv2d(base_channels, 3,kernel_size=3, stride=1, padding=1)
        self.embedded_dim = embedded_dim
        self.time_mlp = nn.Sequential(nn.Linear(embedded_dim,embedded_dim),nn.SiLU(),nn.Linear(embedded_dim,embedded_dim))

        self.norm = nn.GroupNorm(8,base_channels)
        self.silu = nn.SiLU()


        
        encoder_channels = [base_channels,base_channels*2,base_channels*2,base_channels*4]
        num_res_blocks = 2
        attn_levels = [False, True, True, False] #解耦 down blocks
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = encoder_channels[0]  # 从 input_conv 的输出开始
        for i, out_ch in enumerate(encoder_channels):
            level = nn.ModuleList()
            for j in range(num_res_blocks):
                level.append(ResBlock(in_ch, out_ch,embedded_dim))
                in_ch = out_ch  # 后续 ResBlock 用 out_ch → out_ch
            if attn_levels[i]:
                level.append(self_attention(out_ch))
            self.encoder.append(level)
            
            if i < len(encoder_channels) - 1:  # 最后一级不下采样
                self.downsamples.append(Downsample(out_ch))
                
        self.bottleneck = nn.ModuleList([ResBlock(encoder_channels[-1],encoder_channels[-1],embedded_dim),
                                         self_attention(encoder_channels[-1]),
                                         ResBlock(encoder_channels[-1],encoder_channels[-1],embedded_dim)])    

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]
        prev_out_ch = encoder_channels[-1]  
        for i, out_ch in enumerate(decoder_channels):
            in_ch = prev_out_ch + decoder_channels[i]
            level = nn.ModuleList()
            for j in range(num_res_blocks):
                level.append(ResBlock(in_ch, out_ch,embedded_dim))
                in_ch = out_ch
            if attn_levels[i]:
                level.append(self_attention(out_ch))
            self.decoder.append(level)
            
            if i < len(decoder_channels) - 1:
                self.upsamples.append(Upsample(out_ch))
            
            prev_out_ch = out_ch



    def forward(self,x,t):
        "x: (B,3,32,32)"
        t_emb = self.time_mlp(sinusoidal_embedding(self.embedded_dim, t))
        x = self.entry(x)  #变成base channel通道
        skips = [] #(h1,h2,h3,h4)
        for i, level in enumerate(self.encoder):
            for block in level:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        for block in self.bottleneck:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)

        for i, level in enumerate(self.decoder):
            if i > 0:
                x = self.upsamples[i - 1](x)
            x = torch.cat([x, skips.pop()], dim=1)
            for block in level:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)
        x = self.silu(self.norm(x))
        return self.exit(x)


    