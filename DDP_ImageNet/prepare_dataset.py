from torchvision import transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def load_imagenet(args):
    transform = transforms.Compose([ # Transforming and augmenting images
        transforms.RandomHorizontalFlip(), # 랜덤 좌우 반전
        # transforms.RandomVerticalFlip(), # 랜덤 상하 반전
        # transforms.ColorJitter(), # 랜덤 색상필터
        transforms.RandomResizedCrop((args.imgsz, args.imgsz)), # 랜덤으로 리사이즈 후, cropping
        transforms.ToTensor(), # Tensor로 변환
        transforms.Normalize((0.485, 0.465, 0.406), (0.229, 0.224, 0.225)), # image net 정규화 값
    ])
    # 데이터셋 읽기
    train_data = dset.ImageNet('../../../../media/data1/data/Imagenet', split='train', transform=transform)
    test_data = dset.ImageNet('../../../../media/data1/data/Imagenet', split='val', transform=transform)
    return train_data, test_data
   
    
def load_cifar10(args):
    transform = transforms.Compose([ # Transforming and augmenting images
        transforms.RandomHorizontalFlip(), # 랜덤 좌우 반전
        # transforms.RandomVerticalFlip(), # 랜덤 상하 반전
        # transforms.ColorJitter(), # 랜덤 색상필터
        transforms.RandomResizedCrop((args.imgsz, args.imgsz)), # 랜덤으로 리사이즈 후, cropping
        transforms.ToTensor(), # Tensor로 변환
        orms.Normalize((0.485, 0.465, 0.406), (0.229, 0.224, 0.225)), # image net 정규화 값
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # Cifar10 정규화 값
    ])
    # 데이터셋 읽기
    train_data = dset.CIFAR10('../../../../media/data1/kangjun/Cifar10', train=True, download=True, transform=transform) # Cifar10 dataset
    test_data = dset.CIFAR10('../../../../media/data1/kangjun/Cifar10', train=False, download=True, transform=transform) # Cifar10 dataset
    return train_data, test_data
    
def load_mnist(args):
    transform = transforms.Compose([ # Transforming and augmenting images
        # transforms.RandomVerticalFlip(), # 랜덤 상하 반전
        # transforms.ColorJitter(), # 랜덤 색상필터
        transforms.RandomResizedCrop((args.imgsz, args.imgsz)), # 랜덤으로 리사이즈 후, cropping
        transforms.ToTensor(), # Tensor로 변환
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 정규화 값
    ])
    train_data = dset.MNIST('../../../../media/data1/kangjun/MNIST', train=True, download=True, transform=transform) # MNIST dataset
    test_data = dset.MNIST('../../../../media/data1/kangjun/MNIST', train=False, download=True, transform=transform) # MNIST dataset
    return train_data, test_data
    
    

def pre_dset(rank, world_size, args):
    assert not args.batch_size % world_size, "batch size를 world size로 나뉘어져야 함, batch_size: " + str(args.batch_size) + ", world_size: " + str(world_size) + ", return value: " + str(args.batch_size % world_size)
    batch_per_gpu = args.batch_size // world_size # batch_size가 world_size의 몇 배인지(몫을 리턴)
    if rank == 0:
        print(f"{batch_per_gpu} batches per GPU...") # GPU 당 batch size
        
    if args.data == "imagenet":
        train_set, test_set = load_imagenet(args)
    elif args.data == "cifar10":
        train_set, test_set = load_cifar10(args)
    elif args.data == "mnist":
        train_set, test_set = load_mnist(args)
    else:
        print('올바른 dataset이 아닙니다.')
        return None
    
    
    
    # DDP로 분산 처리하기 위한 데이터 할당
    train_sampler = DistributedSampler(train_set, # 데이터셋
                                       num_replicas=world_size, # process 개수
                                       rank=rank, # process ID
                                       shuffle=args.shuffle, # shuffle 여부(DDP는 True, 단일은 False)
                                      )
    test_sampler = DistributedSampler(test_set,
                                      num_replicas=world_size, # process 개수
                                      rank=rank, # process ID
                                      shuffle=False, # Test set은 섞을 필요 없음
                                     )
    
    # DataLoader
    train_loader = DataLoader(train_set, # 데이터 셋
                              batch_size=batch_per_gpu, # GPU에 맞는 batch size(CPU는 그냥 batch_size)
                              pin_memory=args.pin_memory, # CPU에서 GPU의 VRAM으로 데이터를 로드해주기 위한 CPU의 Pinned memory
                              num_workers=args.num_workers, # 데이터 프로세싱 시 CPU 코어 할당량
                              sampler=train_sampler,
                              shuffle=not args.shuffle,  # shuffle옵션의 디폴트는 False이지만 여기서는 유동적으로 변경하였음
                             )
    test_loader = DataLoader(test_set,
                             batch_size=batch_per_gpu,
                             pin_memory=args.pin_memory, # CPU에서 GPU의 VRAM으로 데이터를 로드해주기 위한 CPU의 Pinned memory
                             num_workers=args.num_workers, # 데이터 프로세싱 시 CPU 코어 할당량
                             sampler=test_sampler,
                             shuffle=False
                            )

    return train_loader, test_loader