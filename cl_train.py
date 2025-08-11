import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
import wandb
from data_utils import data_split, MultiTransform, TransformedDataset
from contrastive_loss import ContrastiveLoss
from train_utils import cl_train, cl_epoch_log
from models import CNN

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--views', type=int, default=2, help='number of views')
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')

    parser.add_argument('--proj-dim', type=int, default=128, help='dimension of projection')
    parser.add_argument('--bn', type=bool, default=True, help='batch norm')
    parser.add_argument('--dropouts', nargs='+', type=float, default=[.1, .2, .3, .5])

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
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    _, train_dataset = data_split(dataset, ratio=0.2)
    train_set = TransformedDataset(train_dataset, MultiTransform(transform, args.views))

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    # model
    model = CNN(out_dim=args.proj_dim, batch_norm=args.bn, dropouts=args.dropouts).to(device)
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/backbone.pth'
    
    # train
    criterion = ContrastiveLoss(mode=args.mode, temperature=args.temp)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    wandb.login()
    wandb.init(project='CIFAR-10-Contrastive')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.lf)

    for epoch in range(1, args.epochs + 1):
        train_loss= cl_train(model, train_loader, criterion, optimizer, scheduler, device, epoch, args.lf)
        cl_epoch_log(train_loss, epoch, args.epochs)

    torch.save(model.module.backbone.load_state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print('Finished!')

if __name__ == '__main__':
    main()