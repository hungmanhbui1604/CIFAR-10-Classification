import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
from data_utils import get_hard_negative_dict, save_misclassified
from models import CNN

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dir-name', type=str, default='./misclassified')

    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(mean=-mean/std, std=1/std)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    pil_transform = transforms.Compose([
        denormalize,
        transforms.ToPILImage(),
    ])

    if args.dataset == 'train':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tensor_transform)
    elif args.dataset == 'test':
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tensor_transform)
    classes = dataset.classes
    
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    model = CNN(out_dim=len(classes), batch_norm=ckpt['batch_norm'], dropouts=ckpt['dropouts']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    misclassified = get_hard_negative_dict(model, loader, pil_transform)
    save_misclassified(misclassified, save_dir=args.dir_name)

if __name__ == '__main__':
    main()