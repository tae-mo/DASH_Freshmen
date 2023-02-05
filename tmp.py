import torch
import numpy as np

def Accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0][0]


output = torch.tensor([[0, 0, 0.9, 0.1], [0, 0.85, 0.15, 0], [0, 0, 0.8, 0.2], [1, 0, 0, 0]])
target = torch.tensor([2, 1, 1, 0])
print(Accuracy(output, target))