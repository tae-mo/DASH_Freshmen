from tqdm import tqdm
import torch


def train_fn(train_loader, model, optimizer, criterion, rank):

    model.train()
    loop = tqdm(train_loader, leave=True) #진행
    
    train_loss = []
    correct = []
    count = 0

    for batch_idx, (x, y) in enumerate(loop):
    #for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        train_loss.append(loss.item())
        optimizer.zero_grad() #backprop 전 초기화해서 방향
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(out.data, 1) #텐서의 최대값
        correct.append(int(torch.sum(prediction == y.data)))

        count += x.size(0)

        # update progress bar
        loop.set_postfix(loss=loss.item(), accuracy=100*(sum(correct) / count))

        # if batch_idx % 10 == 0 :
        #     print(f'Batch:{batch_idx} Batch loss:{sum(train_loss)/len(train_loss)} Batch accuracy:{100*(sum(correct) / count):.4f}%')
    
    train_accuracy = sum(correct) / count
    train_loss = sum(train_loss)/len(train_loss)

    return train_loss, train_accuracy
