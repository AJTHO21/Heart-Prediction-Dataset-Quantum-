import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class TrafficDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        """
        Initialize the TrafficDataset.
        
        Args:
            img_dir (str): Directory containing the images
            label_dir (str): Directory containing the labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        # Get all image files
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        
        # Get corresponding label path
        label_path = self.label_dir / f"{os.path.splitext(img_name)[0]}.txt"
        
        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read labels if they exist
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO format: class x_center y_center width height
                    labels.append([float(x) for x in line.strip().split()])
        labels = np.array(labels)
        
        sample = {'image': image, 'labels': labels}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def create_data_loaders(train_dir, valid_dir, test_dir, batch_size=32):
    """
    Create PyTorch data loaders for train, validation and test sets.
    
    Args:
        train_dir (str): Directory containing training data
        valid_dir (str): Directory containing validation data
        test_dir (str): Directory containing test data
        batch_size (int): Batch size for the data loaders
    
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = TrafficDataset(
        os.path.join(train_dir, 'images'),
        os.path.join(train_dir, 'labels')
    )
    
    valid_dataset = TrafficDataset(
        os.path.join(valid_dir, 'images'),
        os.path.join(valid_dir, 'labels')
    )
    
    test_dataset = TrafficDataset(
        os.path.join(test_dir, 'images'),
        os.path.join(test_dir, 'labels')
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, valid_loader, test_loader 