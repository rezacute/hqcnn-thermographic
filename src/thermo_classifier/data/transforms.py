"""Data Transforms Module"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


class ThermalTransforms:
    """Thermal image preprocessing transforms
    
    Provides Albumentations transforms optimized for thermal images.
    """
    
    @staticmethod
    def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """Training transforms with augmentation"""
        return A.Compose([
            A.Resize(*image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(10, 50), 
                p=0.3
            ),
            A.Normalize(
                mean=[0.485], 
                std=[0.229]
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """Validation/test transforms without augmentation"""
        return A.Compose([
            A.Resize(*image_size),
            A.Normalize(
                mean=[0.485], 
                std=[0.229]
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_inference_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """Inference transforms"""
        return ThermalTransforms.get_val_transforms(image_size)
