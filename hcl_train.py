import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
import wandb
from data_utils import data_split, MultiTransform, TransformedDataset, get_hard_negative_dict, HardNegativeContrastiveDataset
from contrastive_loss import HardNegativeContrastiveLoss
from train_utils import hcl_train, cl_epoch_log
from models import CNN

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csf-bs', type=int, default=1024, help='classification batch size')
    parser.add_argument('--views', type=int, default=2, help='number of views')
    parser.add_argument('--hard-negatives', type=int, default=8, help='number of hard negative examples in a batch')
    parser.add_argument('--randoms', type=int, default=54, help='number of random examples in a batch')
    parser.add_argument('--hcl-bs', type=int, default=1024, help='hard negative contrastive learning batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')

    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--backbone-path', type=str, required=True)

    parser.add_argument('--mode', type=str, default='scl', choices=['scl', 'simclr'], help='mode of contrastive loss')
    parser.add_argument('--temp', type=float, default=0.1, help='temperature of contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')

    parser.add_argument('--lf', type=int, default=10, help='log frequency')

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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        tensor_transform
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    classes = dataset.classes
    _, train_dataset = data_split(dataset, ratio=0.2)

    csf_train_set = TransformedDataset(dataset=train_dataset, transform=tensor_transform)
    csf_loader = DataLoader(csf_train_set, batch_size=args.csf_bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist!")
    csf_ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    csf_model = CNN(out_dim=len(classes), batch_norm=csf_ckpt['batch_norm'], dropouts=csf_ckpt['dropouts']).to(device)
    csf_model.load_state_dict(csf_ckpt['model_state_dict'])
    csf_model = nn.DataParallel(csf_model)

    hard_negative_dict = get_hard_negative_dict(csf_model, csf_loader, pil_transform)

    hcl_train_set = HardNegativeContrastiveDataset(dataset=train_dataset,
                                                  positive_transform=MultiTransform(transform=train_transform, views=args.views), 
                                                  hard_negative_transform=train_transform, 
                                                  hard_negatives=args.hard_negatives,
                                                  random_transform=train_transform,
                                                  randoms=args.randoms,
                                                  hard_negative_dict=hard_negative_dict)
    
    hcl_loader = DataLoader(hcl_train_set, batch_size=args.hcl_bs, shuffle=True, num_workers=args.workers, pin_memory=True)

    # model
    if not os.path.exists(args.backbone_path):
        raise FileNotFoundError(f"Backbone path {args.backbone_path} does not exist")
    backbone_ckpt = torch.load(args.backbone_path, map_location=device, weights_only=True)
    hcl_model = CNN(out_dim=backbone_ckpt['projection_dim'], batch_norm=backbone_ckpt['batch_norm'], dropouts=backbone_ckpt['dropouts']).to(device)
    hcl_model.backbone.load_state_dict(backbone_ckpt['backbone_state_dict'])
    hcl_model = nn.DataParallel(hcl_model)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    backbone_path = './ckpts/backbone.pth'

    # train
    criterion = HardNegativeContrastiveLoss(mode=args.mode, temperature=args.temp)
    optimizer = optim.Adam(hcl_model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    wandb.login()
    wandb.init(project='CIFAR-10-Hard-Negative-Contrastive')
    wandb.config.update(args)
    wandb.watch(hcl_model, log="gradients", log_freq=args.lf)

    for epoch in range(1, args.epochs + 1):
        train_loss= hcl_train(hcl_model, hcl_loader, criterion, optimizer, scheduler, device, epoch, args.lf)
        cl_epoch_log(train_loss, epoch, args.epochs)

    torch.save({
        'epoch': args.epochs,
        'projection_dim': backbone_ckpt['projection_dim'],
        'batch_norm': backbone_ckpt['batch_norm'],
        'dropouts': backbone_ckpt['dropouts'],
        'backbone_state_dict': hcl_model.module.backbone.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, backbone_path)
    artifact = wandb.Artifact('ckpt', type='ckpt')
    artifact.add_file(backbone_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print('Finished!')

if __name__ == '__main__':
    main()