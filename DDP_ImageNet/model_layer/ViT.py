import torch
from torch import nn

from einops import rearrange, repeat # 이미지 변환 모듈
from einops.layers.torch import Rearrange # 텐서 변환 모듈

def pair(t) -> tuple: # 값이 하나면 2개의 같은 요소로 반환
    return t if isinstance(t, tuple) else (t, t) #  isinstance: 해당 데이터가 어떤 자료형인지를 확인(return: boolean)

class PreNorm(nn.Module): # Batch Normalization
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim) # 배치 정규화
        self.fn = fn # 각각의 layer(Attention or FeedForward)를 추가
    def forward(self, x, **kwargs): # **kwargs: keword argument의 줄임말로 키워드를 제공({'키워드': '특정값'} 형태로 받아짐)
        return self.fn(self.norm(x), **kwargs) # batch normalize layer 통과 후, 지정한 layer(Attention or FeedForward) 통과
    
class FeedForward(nn.Module): # MLP
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), # FC layer
            nn.GELU(), # activation function
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), # 입력과 같은 모양으로 출력
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module): # Multi-Head Attention module
    def __init__(self, dim, heads=8, dim_head = 64, dropout=0.): # dim: pathch를 벡터화한 길이
        super(Attention, self).__init__()
        inner_dim = dim_head * heads # 입력 벡터 길이 계산
        project_out = not (heads ==1 and dim_head == dim) # projection을 할 필요가 있는지에 대한 여부, head 개수가 1이 아니면서 dim_head와 dim이 같지 않을 경우 True로 반환
        
        self.heads = heads # Multi-Head의 head 개수
        self.scale = dim_head ** -0.5 # head dim의 제곱근으로 나눔(Transformer에선 dim_head가 key벡터 사이즈에 속함)
        
        self.attend = nn.Softmax(dim = -1) # softmax를 적용할 축을 마지막 차원에 적용 => 마지막 차원의 요소 합이 1이 됨(확률로 표현)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # query, key, value를 생성
        self.to_out = nn.Sequential( # q, k, v를 한번에 연산
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out  else nn.Identity() # 만약 project_out이 True면 Linear를, False면 입력 그대로 반환
        
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1) # query, key, value를 생성, chunk: list데이터를 지정된 개수의 리스트로 나누어 묶음
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # query, key, value의 tensor를 연산하기 위해 변환(차원 요소 정렬)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # vector 사이의 score를 구한 후, 더 안정적인 gradient를 가지기 위해 head dim의 제곱근으로 나눠줌
        
        attn = self.attend(dots) # softmax
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v) # softmax * value
        out = rearrange(out, 'b h n d -> b n (h d)') # tensor를 다시 입력과 같이 shape복원(차원 요소 정렬)
        return self.to_out(out)
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([]) # Module을 저장할 리스트(depth만큼 sub-layer를 추가)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), # Attention layer
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)) #FeedForward layer
            ]))
    def forward(self, x):
        for attn, ff in self.layers: # Sub-layer 학습
            x = attn(x) + x # Attention layer 통과
            x = ff(x) + x # FeedForward layer 통과
        return x
        
class ViT(nn.Module):
    def __init__(self, *, image_size=256, patch_size=16, num_classes=1000, dim=1024, depth=12, heads=8, mlp_dim=2048, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size) # image의 height, width
        patch_height, patch_width = pair(patch_size) # patch size
        
        # assert: True가 나와야 다음으로 넘어가짐
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size 이미지 크기를 패치 크기로 나뉘어 떨어져야 합니다!'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) # 이미지를 패치로 나눴을 때 나오는 개수(몫)
        patch_dim = channels * patch_height * patch_width # 채널 수 * patch**2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # Lineary에 넣기 위한 shape을 맞춰줌 {batch_size, image flatten, channel*patch*patch}
            nn.LayerNorm(patch_dim), # 배치 정규화
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim), # 배치 정규화
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # 클래스 토큰(dim만큼 파라미터를 만듦)
        self.dropout = nn.Dropout(emb_dropout) # dropout layer
        
        self. transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool # cls or mean
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential( # MLP Head
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        
    def forward(self, img):
        # Embedded Patches
        x = self.to_patch_embedding(img) # 이미지 임베딩
        b, n, _ = x.shape # batch size와 N을 받음
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # batch 크기만큼 늘림 {batch size, class token, dimension}
        x = torch.cat((cls_token, x), dim=1) # input 데이터 가장 앞에 class token을 붙힘
        x += self.pos_embedding[:, :(n+1)] # input 데이터에 positional 행렬을 더함
        x = self.dropout(x)
        
        # Transformer Encoder
        x = self.transformer(x)
        
        # pool == mean이면 각 행의 평균을 구하여 반환
        # pool == cls이면 각 행의 가장 첫번째 인덱스인 클래스 토큰을 반환
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # tensor.mean(dim=1) => 축소할 차원으로 평균을 구함
        
        x = self.to_latent(x) # Identity layer
        return self.mlp_head(x) # MLP Head