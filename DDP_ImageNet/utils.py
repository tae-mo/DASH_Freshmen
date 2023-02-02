import torch
import os

def save_ckpt(state, file_name="./model_checkpoint/best_model.pth") -> None:
    torch.save(state, file_name)