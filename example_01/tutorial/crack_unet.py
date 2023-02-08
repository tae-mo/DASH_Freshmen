# JW 
# Dragonball
# unet_resnet.py

#torchrun --nproc_per_node 1 /home/jovyan/G1/dragon_ball/unet_resnet.py -b 16 -e 20 -n Unet_resnet-1 -l 0.05

import argparse
import os
from re import M
import csv
from pyrsistent import b
from pathlib import Path
import tqdm
import gc

import torch
import torch.distributed as dist

# Custome Dataset for crack data
from data.Dataset import *

# Metric and logging
from utils.train_utils import *
# from G1.dragon_ball.utils.train_utils import *

from utils.metrics import *
# from G1.dragon_ball.utils.metrics import *

# convolution models for semantic segmentation
from model.unet import *
# from G1.dragon_ball.model.unet import *
from model.FCHardnet import *
import segmentation_models_pytorch as smp
# from model.Hardnet import *

from torch.nn.parallel import DistributedDataParallel as DDP
# Weights & Biases
import wandb


save_dir = '/home/jovyan/G1/dragon_ball/result'

def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--train', '-t', default=True)
    parser.add_argument('--project_name', '-n', default='crack_seg')
    parser.add_argument('--data_loader', '-d', default='mini')
    parser.add_argument('--no_cuda', '-c', default=False)
    parser.add_argument('--model', '-m', default='resnet101')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epoch', '-e', type=int, default=15)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    parser.add_argument('--keep', '-k', default='not_use_best')
    # parser.add_argument('—save_model', '-s', action='store_true', default=False) lr

    return parser.parse_args()

def create_model(device, type ='vgg16'):
    one_class = 1
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=False)
    elif type == 'resnet101':
        encoder_depth = 101
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=one_class, pretrained=False)
    elif type == 'resnet34':
        encoder_depth = 34
        print('create resnet34 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=one_class, pretrained=False)
    elif type == 'example':
        encoder_depth = 34
        print('create vision Transformer model')
        model = smp.PAN(encoder_name='')
    else:
        assert False
    model.eval()
    return model.to(device)

def train(train_loader, model, optimizer, criterion, device, 
          epoch, batch_size):
    model.train()
    if epoch == 0: print("start train {}".format(len(train_loader)))
    jaccards = []
    dices = []
    losses = []

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description(f'Train Epoch {epoch}')
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        # Forward 
        outputs = model(image)
        # loss_func
        loss = criterion(outputs, label)
        losses.append(loss.item())
        # Gradinet 
        loss.backward() 
        optimizer.step() 

        m_j = jaccard_iou(outputs, label, device)
        dice = uj_dice(outputs, label)
        jaccards.append(m_j)
        dices.append(dice)

        tq.set_postfix( loss ='{:.5f}'.format((sum(losses)/len(losses))), 
        jcAcc ='{:.5f}'.format(sum(jaccards)/len(jaccards)), 
        dcAcc ='{:.5f}'.format(sum(dices)/len(dices)))
        tq.update(batch_size)
    tq.close()
    return (sum(jaccards)/len(jaccards)), (sum(dices)/len(jaccards)), (sum(losses)/len(losses)) 

def test(test_loader, model, criterion, device, epoch, batch_size):
    model.eval()

    jaccards = []
    dices = []
    losses = []

    tq = tqdm.tqdm(total=(len(test_loader) * batch_size))
    tq.set_description(f'Test Epoch {epoch}')
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            outputs = model(image) 
            # loss_func 
            loss = criterion(outputs, label)
            losses.append(loss.item())

            # metric_f = torch_iou_fast(outputs, label)
            m_j = jaccard_iou(outputs, label, device)
            jaccards.append(m_j)
            dice = uj_dice(outputs, label)
            dices.append(dice)

            tq.set_postfix(loss='{:.5f}'.format((sum(losses)/len(losses))), 
            jcAcc ='{:.5f}'.format(sum(jaccards)/len(jaccards)), 
            dcAcc ='{:.5f}'.format(sum(dices)/len(dices)))
            tq.update(batch_size) 
    tq.close()
    return (sum(jaccards)/len(losses)), (sum(dices)/len(losses)), (sum(losses)/len(losses))


"""def find_latest_model_path(dir):
model_paths = []
epochs = []
for path in Path(dir).glob('*.pt'):
print(path)
if 'epoch' not in path.stem:
continue
model_paths.append(path)
parts = path.stem.split('_')
epoch = int(parts[-1])
epochs.append(epoch)

if len(epochs) > 0:
epochs = np.array(epochs)
max_idx = np.argmax(epochs)
return model_paths[max_idx]
else:
print("nothing in dir "+dir)
return None"""

def main():
    args = parse_args() 
    torch.manual_seed(args.seed)
    device_id = int(os.environ["LOCAL_RANK"])

    #
    wandb.login()
    wandb.init(project="Unet", entity="kau-aiclops", name=args.project_name)
    wandb.config.update(args) # adds all of the arguments as config variables

    #
    weight_decay = 1e-4
    num_workers = 4
    momentum = 0.9
    epoches = args.epoch # 15
    batch_size=args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_id == 0:
        print(torch.__version__, torch.cuda.__path__, device) 
        print("dist") if dist.is_available() else print("can not use dist")
    #
    latest_model_path = find_latest_model_path(save_dir+'/'+args.project_name)
    path = create_save_dir(name=args.project_name, default_dir=save_dir) # type: ignore
    best_model_path = os.path.join(*[path, 'model_best.pt'])
    logger = get_log(path)

    #
    train_set = CrackDataset(split='train')
    test_set = CrackDataset(split='test')
    train_loader = get_loader(train_set, batch_size=args.batch_size, num_workers=num_workers, mode=args.data_loader)
    test_loader = get_loader(test_set, batch_size=args.batch_size, num_workers=num_workers, mode=args.data_loader)
    #
    model = create_model(device, type=args.model)
    # model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    wandb.watch(model)

    #
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to('cuda')

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        start_epoch = start_epoch

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_test_loss = best_state['test_loss']

        print(f'Restored model at epoch {start_epoch}. Min validation loss so far is : {min_test_loss}')
        start_epoch += 1
        print(f'1 Started training model from epoch {start_epoch}')
    elif args.keep == 'use_best' : 
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(best_model_path)
        min_test_loss = best_state['test_loss']
        start_epoch = 10
    else:
        print('0 Started training model from epoch 0')
        start_epoch = 0
        min_test_loss = 9999



    test_losses = []
    test_acces = []
    for epoch in range(start_epoch, epoches+start_epoch):
        print('\n')

        # train
        trainM, trainD, trainL = train(train_loader, model, optimizer, criterion, device, epoch, batch_size)

        # dist.all_reduce
        #test
        testJ, testD, testL = test(test_loader, model, criterion, device, epoch, batch_size)
        test_losses.append(testL)
        test_acces.append(testJ)

        #log
        log = {"epoch": epoch, "Train Loss":trainL, "Train IOU":trainM, "Train Dice":trainD, 
        "Test Loss":testL, "Test IOU":testJ, "Test Dice":testD }
        logger.info(log)
        wandb.log(log) 

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[path, f'model_epoch_{epoch}.pt'])
        print(epoch_model_path)
        torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'test_loss': testL,
        'train_loss': trainL
        }, epoch_model_path)

        if testL < min_test_loss:
            min_test_loss = testL

            torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'test_loss': testL,
            'train_loss': trainL
            }, best_model_path)

        with open(path+'log_csv.csv','a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(log.keys())
                w.writerow(log.values())
            else:
                w.writerow(log.values())
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    wandb.finish()
