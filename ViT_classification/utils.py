import torch.distributed as dist
import torch

def ddp_setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def _save_snapshot(model, epoch, snapshot_path):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "EPOCHS_RUN": epoch
    }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")


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


