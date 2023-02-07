import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Embedding dim으로 변환하며 패치크기의 커널로 패치크기만큼 이동하여 이미지를 패치로 분할 할 수 있음.


    def forward(self, x): 
        # input x sample size == batch size x shape is n_samples, in_chans, img_size, img_size tensor
        # return shape n_samples, n_patches, embed_dim

        x = self.proj(x)  # (n_samp == batch size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # 세번째 차원부터 끝까지 flatten (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)

        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads #모든 어텐션헤드 연결하면 입력과 동일한 차원의 새로운 텐서 얻음
        self.scale = self.head_dim ** -0.5  # 1 / root(self.head_dim)
        '''
        # 나중에 query와 key를 곱하고 softmax를 취하기전에 scale factor로 나눠주는데 이 scale factor의 역할은 
        query @ key 의 값이 커지게 되면 softmax 함수에서의 기울기 변화가 거의 없는 부분으로 가기때문에 gradient vanishing
        문제를 해결하려면 scaling을 해주어야 한다고 Attention is all you need 논문에서 주장
         
        '''
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) #linear mapping 
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim) #concat된 헤드 가져와서 새로운 공간에 매핑함
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x): #input and output tensors are gonna have a same shape 
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches+1, dim*3)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches+1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각의 n_heads끼리 query, key, value로 나눔
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches+1)  dot product를 위한 transpose
        # dot product를 통해 query와 key사이의 유사도를 구함
        dp = (
            q @ k_t
            ) * self.scale  # (n_samples, n_heads, n_patches+1, n_patches+1)  @: dot product (2x1)@(1x2)=(2x2)
        attn = dp.softmax(dim=-1)  # attention (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches+1, head_dim)
        # 원래 차원으로 되돌림.
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # concat attention heads (n_samples, n_patches+1, dim) 3 dim tensor


        x = self.proj(weighted_avg)  # linear projection (n_samples, n_patches+1, dim)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module): #1 hidden layer 
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU() #Gaussian Error Linear Unit의 약자이며 다른 알고리즘보다 수렴속도가 빠르다는 특징을 가지고 있다.
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  #n_samples , n_patches+1 , hidden_features
        x = self.act(x) #n_samples , n_patches+1 , hidden_features
        x = self.drop(x) #n_samples , n_patches+1 , hidden_features
        x = self.fc2(x) #n_samples , n_patches+1 , hidden_features
        x = self.drop(x) #n_samples , n_patches+1 , hidden_features
        return x

class Block(nn.Module):

    #2 norm layers, 
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)#pretrained model과동일하게 맞춤 
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)  # 3072(MLP size)
        self.mlp = MLP(
            in_features=dim,
            hidden_features= hidden_features,
            out_features=dim,
        )

    def forward(self, x): #residual block 만들어서 original input tensor을 새 텐서(attn,norm)에 더함
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        #2개 LAYER NORM 쓰는이유 : 각자 파라미터 다르게 갖고있어서 
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12, #블록 갯수 
            n_heads=12, #어텐션헤드
            mlp_ratio=4., #mlp모듈의 hidden dimension determine
            qkv_bias=True, #if true include bias to the query, key, value
            p=0., #float dropout prob
            attn_p=0. #same
            ):

        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #1st token of sequence 
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        #cls token 의 위치도 알고싶기 때문에 +1 
        
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)  # 12개의 block
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6) 
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x): #input tensor == batch of images

        """
        x : torch.tensor shape n_samples, in_chans, img_size, img_size

        return 
        logits : torch.tensor n_samples, n_classes

        """

        n_samples = x.shape[0]
        x = self.patch_embed(x)  # turn into  (n_samples, n_patches, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim) why 1??
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1+n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)  # (n_samples, 577, 768)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just tje CLS token
        x = self.head(cls_token_final)

        return x

# def test():
#     BATCH_SIZE = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = Resnet50().to(device)
#     y = net(torch.randn(BATCH_SIZE, 3, 224, 224).to(device)).to(device)
#     assert y.size() == torch.Size([BATCH_SIZE, 1000])
#     print(y.size())


# if __name__ == "__main__":
#     test()
