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
    def __init__(self, local_rank):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank

    def open(self, fp, mode=None):
        if self.local_rank != 0: return
        if mode is None: mode = 'w'
        self.file = open(fp, mode)

    def write(self, msg, is_terminal=1, is_file=1):
        if self.local_rank != 0: return
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1:
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
    
def synset2word(synset_path):
    label_dict = {}
    with open(synset_path, 'r') as f:
        synset_word = f.readlines()
        for i in range(len(synset_word)):
            synset = synset_word[i].split()[0]
            word = re.sub(r'[^a-zA-Z]', '', synset_word[i].split()[1])
            label_dict[synset] = word
            
    return label_dict

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0