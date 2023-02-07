from tqdm import tqdm
import torch


def test_fn(test_loader, model, criterion, rank):
    model.eval()
    loop = tqdm(test_loader, leave=True) #진행률

    test_loss = []
    correct = []
    count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(rank), y.to(rank)
            out = model(x)
            loss = criterion(out, y)
            _, prediction = torch.max(out.data, 1)
            
            #print("data   ", prediction,"real answer   " y)
            test_loss.append(loss.item())
            correct.append(int(torch.sum(prediction == y.data)))
            count += x.size(0)

            # update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=100*(sum(correct) / count))

        test_accuracy = sum(correct) / count
        test_loss = sum(test_loss)/len(test_loss)

        return test_loss, test_accuracy
