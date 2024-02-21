# DASH_Freshmen
### Usage
```
git clone https://github.com/tae-mo/DASH_Freshmen.git
cd DASH_Freshmen
git branch <YOUR NAME> # ex) git branch taejune
git checkout <YOUR NAME> # ex) git checkout taejune

# ~~ do your things ~~ #

git add . 
git commit -m "<WHAT YOU DID>" # ex) git commit -m "resnet from scratch"
git push origin <YOUR BRANCH NAME> # ex) git push origin taejune
```

### Code Execution
- ResNet
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 main.py --exp './ResNet50_image_net' --model resnet --data imagenet --local_rank 1 --learning_rate 1e-4 --epochs 200 --batch_size 512 --every 500 --num_workers 8 --pin_memory --shuffle --imgsz 224

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 main.py --exp './ResNet50_ image_net' --model resnet --data mnist --local_rank 1 --learning_rate 7e-3 --epochs 50 --batch_size 64 --every 1000 --num_workers 8 --pin_memory --shuffle --is_wandb
```
- ViT
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 main.py --exp './ViT_Cifar10' --model vit --data cifar10 --learning_rate 3e-5 --epochs 100 --batch_size 126 --every 500 --gamma 0.5 --num_workers 8 --pin_memory --shuffle --imgsz 224 --is_wandb
```
- Xception
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 main.py --exp './Xception' --model xception --data DeepFake --learning_rate 1e-3 --epochs 20 --batch_size 256 --every 500 --gamma 0.7 --num_workers 8 --pin_memory --shuffle --imgsz 250 --is_wandb
```

### NOTE:
1. BZNet: Dash-9 /media/data1/sangyup/BZNet-main.zip
2. PUSH IN YOUR OWN BRANCH (not main branch)