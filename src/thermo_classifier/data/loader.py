"""Data Loader Module"""

from torch.utils.data import DataLoader
from thermo_classifier.data.dataset import ThermalDataset
from typing import Optional, Tuple, Union
from pathlib import Path


class ThermalDataLoader:
    """DataLoader factory for thermal datasets"""
    
    @staticmethod
    def create(
        data_path: Union[str, Path],
        domain: str = 'medical',
        split: str = 'train',
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: Optional[bool] = None,
        image_size: Tuple[int, int] = (224, 224),
        transform=None
    ) -> DataLoader:
        """Create DataLoader for thermal dataset
        
        Args:
            data_path: Root path to dataset
            domain: 'medical' or 'industrial'
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle: Whether to shuffle (defaults to True for train)
            image_size: Target image size (H, W)
            transform: Optional custom transform
            
        Returns:
            DataLoader instance
        """
        dataset = ThermalDataset(
            data_path=data_path,
            domain=domain,
            split=split,
            image_size=image_size,
            transform=transform
        )
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    @staticmethod
    def create_from_config(
        data_path: Union[str, Path],
        domain: str = 'medical',
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224)
    ) -> dict:
        """Create all splits (train, val, test)"""
        return {
            'train': ThermalDataLoader.create(
                data_path, domain, 'train', batch_size, num_workers, image_size=image_size
            ),
            'val': ThermalDataLoader.create(
                data_path, domain, 'val', batch_size, num_workers, image_size=image_size
            ),
            'test': ThermalDataLoader.create(
                data_path, domain, 'test', batch_size, num_workers, image_size=image_size
            ),
        }
