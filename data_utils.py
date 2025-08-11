import torch
from torch.utils.data import Dataset, random_split
    
def data_split(dataset, ratio):
    size0 = int(ratio * len(dataset))
    size1 = len(dataset) - size0
    generator = torch.Generator().manual_seed(42)
    dataset0, dataset1 = random_split(dataset, [size0, size1], generator=generator)
    return dataset0, dataset1

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y 
    
class MultiTransform:
    def __init__(self, transform, views):
        self.transform = transform
        self.views = views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.views)]