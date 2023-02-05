
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
import torch

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

def make_patches(img, p):
    patches = np.array([])
    _, h, w = img.shape
    
    for y in range(0, h, p):
        for x in range(0, w, p):
            if (h-y) < p or (w-x) < p:
                break
            
            tiles = img[:, y:y+p, x:x+p]
            
            if patches.size == 0:
                patches = tiles.reshape(1,-1)
                
            else:
                patches = np.vstack([patches, tiles.reshape(1, -1)])
    
    return patches

# make input with patch embedding and positional embedding
class PatchEmbeddings(nn.Module):
    
    def __init__(self, in_channels, patch_size, emb_size, img_size = 224):
        super(PatchEmbeddings, self).__init__()
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = patch_size, stride = patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
    def forward(self, x):
        # x             b, 3, 224, 224
        b, _, _, _ = x.shape
        x = self.projection(x)
        # x             b, 196, 768
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        # cls_token     b, 1, 768
        x = torch.cat([cls_tokens, x], dim=1)
        # x             b, 197, 768
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(dropout)

        self.qkv = nn.Linear(emb_size, emb_size*3)

        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # q, k, v 한번에 연산
        # x             b, 197, 768
        # qkv(x)        b, 197, 768*3
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv=3)
        # qkv           3, b, 8, 197, 96
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # queries       b, 8, 197, 96
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # energy        b, 8, 197, 197

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # sqrt(k)
        scaling = self.emb_size ** (1/2)
        
        att = F.softmax(energy/scaling, dim=-1)
        att = self.att_drop(att)
        # att           b, 8, 197, 197
        
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        
        # out           b, 8, 197, 96
        out = rearrange(out, "b h n d -> b n (h d)")
        # out           b, 197, 768
        out = self.projection(out)
        return out




# fn을 저장하고 들어오는 x를 더해서 forward
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# 
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion*emb_size, emb_size),
        )



class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
                )
            ),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
                )
            )
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])



# 
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__(
            # 8 197 768
            Reduce('b n e -> b e', reduction='mean'),
            # 8 768   n을 평균내서 차원축소
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
            # 8 1000
        )


class ViT(nn.Sequential):
    def __init__(self, in_channels, patch_size, emb_size, img_size, depth, n_classes, **kwargs):
        super().__init__(
            PatchEmbeddings(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# model = ViT(3, 16, 768, 224, 12, 1000)
# input_x = torch.randn(8, 3, 224, 224)
# print(model(input_x).shape)

# print(summary(model, (3, 224, 224), device='cpu'))



