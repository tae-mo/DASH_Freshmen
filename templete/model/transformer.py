import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, embedding_vectors):
        super(Embedder, self).__init__()
        
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=embedding_vectors, freeze=True
        ) # freeze=Ture에 의해 역전파로 갱신되지 않고 변하지 않는다. 
        
    def forward(self, x):
        x_vec = self.embeddings(x)
        
        return x_vec
#

