from utils import AverageMeter, Logger, print_args, save_args, load_args, Accuracy
import time
import tqdm
import torch

def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    model.train()

    for i, (input, target) in enumerate(train_loader):
        
        # 또한 텐서 및 스토리지를 고정하면 비동기(asynchronous) GPU 복사본을 사용할 수 있습니다.
        # 비동기식으로 GPU에 데이터 전달 기능을 추가하려면 non_blocking = True 인수를 to() 또는 cuda() 호출 시 argument로 전달하면 됩니다.
        input = input.to(args.local_rank, non_blocking=True)
        target = target.to(args.local_rank, non_blocking=True)
        
        output = model(input)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc1 = Accuracy(output.data, target, topk=(1,)) # size 1 tensor list
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc1[0], input.size(0))
        
        if 1 & args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val[0]:.4f} ({top1.avg[0]:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), loss=losses, top1=accuracy))
            
    print('* Epoch: [{0}/{1}]\t Train Accuracy: {top1.avg[0]:.3f} Train Loss: {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=accuracy, loss=losses))

    return losses.avg, accuracy.avg

def validate(val_loader, model, criterion, args):
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(args.local_rank)
            target = target.to(args.local_rank)
            
            output = model(input)
            loss = criterion(output, target)
            
            acc1 = Accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            accuracy.update(acc1[0], input.size(0))

            if i % args.print_freq == 0:
                print('Test (on val set): [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top 1-acc {top1.val[0]:.4f} ({top1.avg[0]:.4f})'.format(
                    i, len(val_loader), loss=losses, top1=accuracy))
        print('* Validation Accuracy: {top1.avg[0]:.3f}  Test Loss: {loss.avg:.3f}'.format(
            top1=accuracy, loss=losses))
        return losses.avg, accuracy.avg