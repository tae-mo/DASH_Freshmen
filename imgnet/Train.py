from utils import AverageMeter, Logger, print_args, save_args, load_args, Accuracy
import time
import tqdm
import torch

def train(train_loader, model, criterion, optimizer, epoch, rank, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    
    model.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # 또한 텐서 및 스토리지를 고정하면 비동기(asynchronous) GPU 복사본을 사용할 수 있습니다.
        # 비동기식으로 GPU에 데이터 전달 기능을 추가하려면 non_blocking = True 인수를 to() 또는 cuda() 호출 시 argument로 전달하면 됩니다.
        input = input.to(rank, non_blocking=True)
        target = target.to(rank, non_blocking=True)
        
        output = model(input)
        loss = criterion(output, target)
        
        acc1 = Accuracy(output.data, target, topk=(1,)) # size 1 tensor list
        losses.update(loss.item(), input.size(0))
        top1_acc.update(acc1[0], input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if 1 & args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val[0]:.4f} ({top1.avg[0]:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1_acc))
            
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg[0]:.3f} Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1_acc, loss=losses))

    return losses.avg, top1_acc.avg

def validate(val_loader, model, criterion, rank, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    
    model.eval()
    
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(rank)
            target = target.to(rank)
            
            output = model(input)
            loss = criterion(output, target)
            
            acc1 = Accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1_acc.update(acc1[0], input.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if 1 & args.print_freq == 0:
                print('Test (on val set): [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top 1-acc {top1.val[0]:.4f} ({top1.avg[0]:.4f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1_acc))
        print('* Top 1-err {top1.avg[0]:.3f}  Test Loss {loss.avg:.3f}'.format(
            top1=top1_acc, loss=losses))
        return top1_acc.avg, losses.avg