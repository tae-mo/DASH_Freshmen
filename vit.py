
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
import torch

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

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
    def __init__(self, emb_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(dropout)

        self.qkv = nn.Linear(emb_size, emb_size*3)


    def forward(self, x, mask):
        # q, k, v 한번에 연산
        # x             b, 197, 768
        # qkv(x)        b, 197, 768*3
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv=3)
        # qkv           3, b, 8, 197, 96
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        #queries        b, 8, 197, 96
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # sqrt(k)
        scaling = self.emb_size ** (1/2)
        
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

