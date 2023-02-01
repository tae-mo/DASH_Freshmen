import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Imagenet Training")
    
    ## Config
    parser.add_argument("--exp", type=str, default="./exp/default")
    parser.add_argument("--ddp", type=str, default="False")

    ## training
    parser.add_argument("--data_path", type=str, default="/home/data/imagenet")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valid_iter", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    
    ## data loader
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--num_workers", type=int, default=2) # may cause a bottleneck if set to be 0
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--imgsz", type=int, default=600)
    
    parser.add_argument("--local_rank", type=int, default=0)
    
    return parser.parse_args()

