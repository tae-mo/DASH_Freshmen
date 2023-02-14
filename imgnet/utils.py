import os
import sys
import argparse
import json
import torch
import re
import builtins
import datetime
import torch.distributed as dist

class Logger(object):
    def __init__(self, local_rank=0, no_save=False):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
        self.no_save = no_save
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        if self.local_rank and not self.no_save == 0: self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if self.local_rank == 0:
            if '\r' in msg: is_file = 0
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1 and not self.no_save:
                self.file.write(msg)
                self.file.flush()
    def flush(self): 
        pass
    
def print_args(args, logger=None):
    if logger is not None:
        logger.write("#### configurations ####")
    for k, v in vars(args).items():
        if logger is not None:
            logger.write('{}: {}\n'.format(k, v))
        else:
            print('{}: {}'.format(k, v))
    if logger is not None:
        logger.write("########################")
      
def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
        
def load_args(from_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    return args    

class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
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
        return res