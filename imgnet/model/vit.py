import torch
import torch.nn as nn

from einops import rearrange, repeat, reduce
'''
rearrange doesn't change number of elements and covers different numpy functions (like transpose, reshape, stack, concatenate, squeeze and expand_dims)
rearrange : string으로 transpose, compose, decompose와 같은 shape 변화를 줄 수 있음

reduce combines same reordering syntax with reductions (mean, min, max, sum, prod, and any others)
reduce : string으로 mean, min, max, sum, prod같은 작업을 할 수 있음.

repeat additionally covers repeating and tiling
repeat : string으로 repeat할 수 있음
'''
from einops.layers.torch import Rearrange # torch에서 layer로 사용하게 함

def make_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

class PreNorm(nn.Module): # 모든 블록 전에 먼저 Norm을 하고 attention 수행함.
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # 3(Data 갯수)X6(Feature 갯수) 기준으로 LayerNorm(데이터 샘플 단위) 3개 나옴 VS BatchNorm(특성 단위) 6개 나옴
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # multi head를 내부에서 한 번에 작업하기 위함
        project_out = not (heads == 1 and dim_head == dim) # multi head attention 인지 아닌지
        
        self.heads = heads
        self.scale = dim_head ** -0.5 # root(dim_head)로 나눠 줄 때 필요
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) # input을 q, k, v로 만들기
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # single head일 때는 할 필요 없으므로
        
    def forward(self, x): # [batch_size, seq_len, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1) # chunk으로 세 덩어리로 나눔
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # [batch_size, heads, seq_len, dim_head] X 3
        
        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        
        attn = self.attend(energy)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v) # [batch_size, heads, seq_len, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out) # [batch_size, seq_len, emb_dim]
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # 네트워크가 깊어질 수록 adaptive dropout 형태로 입력치에 가중치를 부여야여 zoneout?? 되는 형태 // ReLU에서 0근처에 살짝 볼록해짐 => 모든 점에서 미분가능해
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x # skip connection 필요함
            x = ff(x) + x
        return x # [batch_size, seq_len, emb_dim]
 
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=7, mlp_dim=3072, dim_head=64, dropout=0., pool='cls', channels=3, emb_dropout=0.):
        super().__init__()
        image_height, image_width = make_tuple(image_size)
        patch_height, patch_width = make_tuple(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image sizes must be divisible by the patch size'
        
        num_patchs = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_width * patch_height
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), # [batch_size, patch_num, patch_dim]
            nn.Linear(patch_dim, dim) # [batch_size, patch_num, emb_dim]
        )
        
        
        # 왜 아래가 퍼포먼스 상향이 있다고 저자는 말했다는 찌라시가 있을까?
        # self.to_patch_embedding = nn.Sequential(
        #     # using a conv layer instead of a linear one -> performance gains
        #     nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size), # [batch_size, emb_dim, (H/path_size), (W/path_size)]
        #     Rearrange('b e (h) (w) -> b (h w) e'), # [batch_size, patch_num, emb_dim]
        # )
        
        
        self.pos_embedding = nn.Parameter(torch.rand(1, num_patchs + 1, dim)) # nostion에 대한 정보인데 rand를 써도 괜찮을까? 학습이 되면서 찾아지나?
        self.cls_token = nn.Parameter(torch.rand(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        # self.to_latent = nn.Identity() BYOL을 사용할거 아니면 필요없음 lucidrain피셜
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), # 모든 block 전에 LN사용함.
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        x = self.to_patch_embedding(x)
        batch_size, num_patchs, emb_dim = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size) # 맨 앞에 붙일 Class 토큰 만들기
        x = torch.cat((cls_tokens, x), dim=1) # 시퀀스 맨 앞에 붙이기 [batch_size, patch_num + 1, emb_dim]
        x += self.pos_embedding[:, :(num_patchs+1)] # positional embedding 더하기
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        # mlp head에 들어갈 때 맨 앞 cls 토큰만 가져갈지 전체를 평균내어 가져갈지 결정하기
        x = x[:, 0] if self.pool == 'cls' else reduce(x, 'b n d -> b d', 'mean') # [batch_size, emb_dim]
        
        # x = self.to_latent(x)
        
        return self.mlp_head(x) # [batch_size, num_classes]