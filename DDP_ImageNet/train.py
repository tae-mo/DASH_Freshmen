import torch
import time
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, rank, epoch, args, train_len) -> (float, float):
    model.train() # 학습을 위한 train mode로 변경
    running_loss, logging_loss, train_acc, logging_acc = 0, 0, 0, 0 # 학습 loss, 로그 출력용 loss, 학습 Accuracy, 로그 출력용 loss,
    # tq = tqdm(total=(len(train_loader) * args.batch_size))
    tq = tqdm(total=train_len)
    tq.set_description(f'Train Epoch {epoch}')
    for idx, (data, label) in enumerate(train_loader):
        if args.pin_memory: # 고정된 메모리
            data, label = data.to(rank, non_blocking=True), label.to(rank, non_blocking=True) # non_blocking: 비동기로 GPU에 객체를 전달
        else:
            data, label = datda.to(rank), label.to(rank) # process Id만 지정
    
        out = model(data) # 모델에 데이터 넣기
        loss = criterion(out, label) # loss 함수로 입력과 라벨의 loss 구하기
        
        running_loss += loss.item()
        logging_loss += loss.item()
        train_acc += (out.detach().argmax(-1) == label).float().sum() / len(data) # accuracy
        logging_acc += (out.detach().argmax(-1) == label).float().sum() / len(data) # accuracy
        
        if rank == 0 and args.every != -1:
            if not idx % args.every: # 정해진 수마다 출력 (나눈 값의 나머지가 0이면 출력됨)
                print(f"[{idx}/{len(train_loader)}] loss per {args.every}: {logging_loss / args.every}, accuracy per {args.every}: {logging_acc / args.every}")
                logging_loss = 0
                logging_acc = 0
        
        tq.set_postfix(Loss ='{:.5f}'.format(running_loss/(idx+1)), Acc ='{:.5f}'.format(train_acc/(idx+1)))
        tq.update(args.batch_size) # can not get it...
        
        optimizer.zero_grad() # 학습마다 기울기를 0으로 초기화
        loss.backward() # 가중치와 편향을 계산
        optimizer.step() # 가중치 업데이트
    tq.close()
    
    return running_loss / len(train_loader), train_acc / len(train_loader) # loss 및 acc 리턴

def validate(model, val_loader, criterion, rank, args):
    model.eval() # 검증 시에는 Batch Normalization과 Dropout를 학습과 다르게 설정해야 함
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(rank), label.to(rank)
            
            out = model(data) # 모델에 데이터 넣기
            loss = criterion(out, label) # loss 함수로 입력과 라벨의 loss 구하기
            
            val_acc += (out.detach().argmax(-1) == label).float().sum() / len(data) # val accuracy
            val_loss += loss # val loss
            
        return val_acc / len(val_loader), val_loss / len(val_loader) # val loss 및 val acc 리턴