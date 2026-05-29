"""
Thermal Dataset
================
PyTorch dataset for thermal images.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Union, Callable
import numpy as np
from PIL import Image
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ThermalDataset(Dataset):
    """Thermal Image Dataset
    
    Supports both Medical and Industrial domain thermal images.
    
    Directory Structure:
        data_path/
            train/
                class1/
                    img1.png
                    img2.png
                class2/
            val/
                class1/
                class2/
            test/
                class1/
                class2/
    
    Args:
        data_path: Root path to dataset
        domain: 'medical' or 'industrial'
        split: 'train', 'val', or 'test'
        transform: Optional albumentations transform
        image_size: Target image size (H, W)
    """
    
    DOMAIN_CLASSES = {
        'medical': ['normal', 'benign', 'malignant'],
        'industrial': ['normal', 'overheating', 'electrical_fault', 'structural_defect', 'thermal_bridge', 'moisture']
    }
    
    def __init__(
        self,
        data_path: Union[str, Path],
        domain: str = 'medical',
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_path = Path(data_path)
        self.domain = domain
        self.split = split
        self.image_size = image_size
        self.classes = self.DOMAIN_CLASSES[domain]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Default transforms if none provided
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform
        
        # Load samples
        self.samples = self._load_samples()
        
    def _get_default_transform(self, split: str) -> A.Compose:
        """Get default transforms for split"""
        if split == 'train':
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),  # Thermal single channel
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all sample paths and labels"""
        samples = []
        
        split_path = self.data_path / self.split
        if not split_path.exists():
            # Try without split (flat structure)
            split_path = self.data_path
        
        for class_name in self.classes:
            class_path = split_path / class_name
            if class_path.exists():
                for img_path in class_path.glob('*.png'):
                    samples.append((img_path, self.class_to_idx[class_name]))
                for img_path in class_path.glob('*.jpg'):
                    samples.append((img_path, self.class_to_idx[class_name]))
                for img_path in class_path.glob('*.jpeg'):
                    samples.append((img_path, self.class_to_idx[class_name]))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('L'))  # Grayscale
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """Get sample counts per class"""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


class ThermalDataLoader:
    """DataLoader wrapper for thermal datasets"""
    
    @staticmethod
    def create(
        data_path: Union[str, Path],
        domain: str = 'medical',
        split: str = 'train',
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: Optional[bool] = None,
        image_size: Tuple[int, int] = (224, 224)
    ) -> DataLoader:
        """Create DataLoader for thermal dataset"""
        
        dataset = ThermalDataset(
            data_path=data_path,
            domain=domain,
            split=split,
            image_size=image_size
        )
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def create_synthetic_dataset(
    output_path: Union[str, Path],
    domain: str = 'medical',
    samples_per_class: int = 50,
    image_size: Tuple[int, int] = (224, 224)
):
    """Create synthetic thermal dataset for testing
    
    Generates random thermal-like images for each class.
    """
    import cv2
    
    output_path = Path(output_path)
    classes = ThermalDataset.DOMAIN_CLASSES[domain]
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            cls_dir = output_path / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            n_samples = samples_per_class if split == 'train' else samples_per_class // 5
            
            for i in range(n_samples):
                # Generate synthetic thermal image
                # Use Perlin noise-like pattern for realistic look
                img = np.random.randn(*image_size) * 30 + 128
                
                # Add some structure
                if 'overheating' in cls or 'malignant' in cls or 'inflammatory' in cls:
                    # Hot spots
                    y, x = np.ogrid[:image_size[0], :image_size[1]]
                    center = (image_size[0]//2, image_size[1]//2)
                    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
                    hot_spot = np.exp(-r/50) * 100
                    img = img + hot_spot
                
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Save
                cv2.imwrite(str(cls_dir / f'{cls}_{i:04d}.png'), img)
    
    print(f"Created synthetic dataset at {output_path}")
    print(f"Domain: {domain}")
    print(f"Classes: {classes}")
