from __future__ import print_function # 파이썬 2와 3 어떤 버젼을 돌리던 모두 파이썬 3 문법인 print() 을 통해 콘솔에 출력이 가능하다.

import matplotlib.pyplot as plt
from torchsummary import summary                    # Successfully installed torchsummary-1.5.1


import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

# 하지만 실제의 ViT에서는 einops같은 Linear Embedding이 아니라 kernal size와 stride size를 patch size로 갖는 Convolutional 2D Layer를 이용한 후 flatten 시켜줍니다. 이렇게 하면 performance gain이 있다고 저자는 말합니다.
from einops import rearrange, reduce, repeat        # Successfully installed einops-0.6.0
from einops.layers.torch import Rearrange, Reduce   # usage: https://velog.io/@lse7530/einops-%EC%9D%B4%EC%9A%A9-%EC%9E%90%EB%A5%B4%EA%B8%B0

class PatchEmbedding(nn.Module): # https://github.com/FrancescoSaverioZuppichini/ViT
    # x: torch.Size(bt, ch, wd, ht) >patch로 바꿔주기> (bt, ch, N, p, p, c) <<<<< BATCHxNx(P*P*C)의 벡터로 임베딩 /// N은 패치의 개수(H*W / (P*P))
    # embedding_size = ch*P*P : 3*16*16=768
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), # channel=3, embedding=3*16*16=768, kernerl=patch=16, stride=patch=16 ?왜 stride를 주는가 patch로 나누기 위해서?
            Rearrange('b e (h) (w) -> b (h w) e'), 
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size)) # N = (img_size // patch_size)**2
        
    def forward(self, x: Tensor) -> Tensor:                         # Embedding 시 class token과 positional embedding을 추가함.
        b, _, _, _ = x.shape
        x = self.projection(x)                                      # 이미지를 패치사이즈로 나누고 flatten
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # 배치마다 맨 앞에 붙을 텐서 >> (b,   1, emb_size)의 파라미터로 생성
        # prepend the cls token to the input                                                            |
        x = torch.cat([cls_tokens, x], dim=1)                       # x에 cls_tokens 붙여주기   >> (b, N+1, emb_size)
        # add position embedding
        x += self.positions

        return x
# end of PatchEmedding class

# Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)    #  a single matrix to compute in one shot queries, keys and values.
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        # self.scaling = (self.emb_size // num_heads) ** -0.5
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    

#  a nice wrapper to perform the residual addition
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

#The attention's output is passed to a fully connected layer 
class FeedForwardBlock(nn.Sequential):  # that upsample by a factor of expansion the input
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        ) 
        # nn.Module 대신에 nn.Sequential을 subclass하면 어떤 이점이 있을까요??
        # 바로 forward method를 작성하지 않아도 됩니다 ㅎㅎ
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))        
        
class ViTscratch(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )