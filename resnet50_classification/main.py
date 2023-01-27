
import os 

import torch 

# GPU Device 

gpu_id = '2' 

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) 

use_cuda = torch.cuda.is_available() 

print("GPU device " , use_cuda) 

device = torch.device('cuda' if use_cuda else 'cpu')