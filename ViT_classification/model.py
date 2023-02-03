import torch
import torch.nn as nn





# def test():
#     BATCH_SIZE = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = Resnet50().to(device)
#     y = net(torch.randn(BATCH_SIZE, 3, 224, 224).to(device)).to(device)
#     assert y.size() == torch.Size([BATCH_SIZE, 1000])
#     print(y.size())


# if __name__ == "__main__":
#     test()
