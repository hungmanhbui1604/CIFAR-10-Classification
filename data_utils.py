import torch
from torch.utils.data import Dataset, random_split
import tqdm
import os
    
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
    

def get_hard_negative_dict(model, loader, pil_transform):
    model.eval()

    misclassified = []

    for images, labels in tqdm.tqdm(loader):
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)

            mis_indices = (preds != labels).nonzero(as_tuple=True)[0]

            for i in mis_indices:
                misclassified.append((images[i], labels[i], preds[i]))  # (img, true_label, predicted_label)

    hard_negative_dict = {}
    for img, true_label, pred_label in misclassified:
        if pred_label.item() not in hard_negative_dict:
            hard_negative_dict[pred_label.item()] = []
        hard_negative_dict[pred_label.item()].append((pil_transform(img), true_label.item()))

    return hard_negative_dict

class HardNegativeContrastiveDataset(Dataset):
    def __init__(self, dataset, positive_transform, hard_negative_transform, hard_negatives, random_transform, randoms, hard_negative_dict):
        self.dataset = dataset
        self.positive_transform = positive_transform
        self.hard_negative_transform = hard_negative_transform
        self.hard_negatives = hard_negatives
        self.random_transform = random_transform
        self.randoms = randoms
        self.hard_negative_dict = hard_negative_dict

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # positve
        anchor_image, anchor_label = self.dataset[idx]
        positive_images = self.positive_transform(anchor_image)
        positive_images = torch.stack(positive_images, dim=0)
        positive_labels = torch.tensor([anchor_label]).repeat(self.positive_transform.views)

        # hard negative
        all_hard_negative_images = self.hard_negative_dict.get(anchor_label, [])
        hard_negative_images = []
        hard_negative_labels = []
        if len(all_hard_negative_images) >= self.hard_negatives:
            hard_negative_indices = torch.randperm(len(all_hard_negative_images))[:self.hard_negatives]
            for i in hard_negative_indices:
                image, label = all_hard_negative_images[i]
                hard_negative_images.append(self.hard_negative_transform(image))
                hard_negative_labels.append(label)
        else:
            for i in range(len(all_hard_negative_images)):
                image, label = all_hard_negative_images[i]
                hard_negative_images.append(self.hard_negative_transform(image))
                hard_negative_labels.append(label)

            remaining_hard_negatives = self.hard_negatives - len(all_hard_negative_images)
            negative_indices = torch.randperm(len(self.dataset))
            cnt = 0
            for i in negative_indices:
                image, label = self.dataset[i]
                if label == anchor_label:
                    continue
                hard_negative_images.append(self.random_transform(image))
                hard_negative_labels.append(label)
                cnt += 1
                if cnt == remaining_hard_negatives:
                    break
        hard_negative_images = torch.stack(hard_negative_images, dim=0)
        hard_negative_labels = torch.tensor(hard_negative_labels)

        # random
        random_indices = torch.randperm(len(self.dataset))[:self.randoms]
        random_images = []
        random_labels = []
        for i in random_indices:
            image, label = self.dataset[i]
            random_images.append(self.random_transform(image))
            random_labels.append(label)
        random_images = torch.stack(random_images, dim=0)
        random_labels = torch.tensor(random_labels)

        x = torch.cat([positive_images, hard_negative_images, random_images], dim=0)
        y = torch.cat([positive_labels, hard_negative_labels, random_labels], dim=0)
        return x, y

def save_misclassified(misclassified, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for pred_label, images_true_labels in misclassified.items():
        dir_name = f"{classes[pred_label]}"
        dir_path = os.path.join(save_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        count = 0
        for image, true_label in images_true_labels:
            image_path = os.path.join(dir_path, f"{classes[true_label]}_{count}.png")
            image.save(image_path)
            count += 1
            
        print(f"Saved {count} images to {dir_path}")