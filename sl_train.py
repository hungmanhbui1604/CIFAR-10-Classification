import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
import wandb
from data_utils import data_split, TransformedDataset
from train_utils import EarlyStopper, sl_train, sl_validate, sl_epoch_log
from models import CNN

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')

    parser.add_argument('--bn', type=bool, default=True, help='batch norm')
    parser.add_argument('--dropouts', nargs='+', type=float, default=[.1, .2, .3, .5])
    parser.add_argument('--backbone-path', type=str, default='')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')

    parser.add_argument('--lf', type=int, default=10, help='log frequency')

    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        val_transform
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    classes = dataset.classes

    val_dataset, train_dataset = data_split(dataset, ratio=0.2)
    train_set = TransformedDataset(train_dataset, train_transform)
    val_set = TransformedDataset(val_dataset, val_transform)

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # model
    model_ckpt = {
        'batch_norm': None,
        'dropouts': None,
        'model_state_dict': None
    }
    if args.backbone_path:
        if os.path.exists(args.backbone_path):
            backbone_ckpt = torch.load(args.backbone_path, map_location=device, weights_only=True)
            model = CNN(out_dim=len(classes), batch_norm=backbone_ckpt['batch_norm'], dropouts=backbone_ckpt['dropouts']).to(device)
            model.backbone.load_state_dict(backbone_ckpt['backbone_state_dict'])

            model_ckpt['batch_norm'] = backbone_ckpt['batch_norm']
            model_ckpt['dropouts'] = backbone_ckpt['dropouts']
        else:
            raise FileNotFoundError(f"Backbone path {args.backbone_path} does not exist")
    else:
        model = CNN(out_dim=len(classes), batch_norm=args.bn, dropouts=args.dropouts).to(device)

        model_ckpt['batch_norm'] = args.bn
        model_ckpt['dropouts'] = args.dropouts
    model = nn.DataParallel(model)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    model_path = f'./ckpts/model.pth'
    
    # train
    criterion = nn.CrossEntropyLoss()
    if args.backbone_path:
        for parameters in model.module.backbone.parameters():
            parameters.requires_grad = False
        optimizer = optim.Adam(model.module.fc.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    early_stopper = EarlyStopper(model, model_ckpt, patience=5, min_delta=0.3)

    wandb.login()
    wandb.init(project='CIFAR-10-Classification')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.lf)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = sl_train(model, train_loader, criterion, optimizer, scheduler, device, epoch, args.lf)
        val_loss, val_acc = sl_validate(model, val_loader, criterion, device, epoch)
        sl_epoch_log(train_loss, train_acc, val_loss, val_acc, epoch, args.epochs)

        if early_stopper.early_stop(val_loss):
            print(f'Training early stopped at epoch {epoch}')
            break
    
    torch.save(model_ckpt, model_path)
    artifact = wandb.Artifact('ckpt', type='ckpt')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print('Traing finished')

if __name__ == '__main__':
    main()