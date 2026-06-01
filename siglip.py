"""
SigLIP2 So400m Loader
----------------------
Loads SigLIP2 So400m from HuggingFace.
Zero-shot model — no training required.
Provides model and dataloaders ready for embedding extraction.

Model      : SigLIP2 So400m
Source     : google/siglip2-so400m-patch14-384
Embedding  : 1152 dimensions
Input size : 384x384
"""

import os
import torch
from transformers import AutoModel, AutoProcessor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CONFIG 
DATA_DIR       = '/your/path/coco'
MODEL_NAME     = 'google/siglip2-so400m-patch14-384'
EMBEDDING_DIM  = 1152
BATCH_SIZE     = 8     # smaller batch due to large model size (400M params)
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'}

# Download model to local drive instead of home directory
os.environ['HF_HOME'] = '/your/path/coco/hf_cache'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# DATASET 
class FilteredImageFolder(ImageFolder):
    # Excludes system folders from dataset
    def find_classes(self, directory):
        classes = [
            f for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))
            and f not in IGNORE_FOLDERS
        ]
        classes.sort()
        return classes, {cls: idx for idx, cls in enumerate(classes)}


class SigLIP2Dataset(torch.utils.data.Dataset):
    # Custom dataset that applies SigLIP2 processor for preprocessing
    # SigLIP2 requires its own processor — standard transforms not compatible
    def __init__(self, subset, processor):
        self.subset    = subset
        self.processor = processor

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            image, label = self.subset[idx]
            # processor resizes to 384x384 and normalises internally
            inputs       = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, label
        except Exception:
            return None


def collate_siglip2(batch):
    # Custom collate for SigLIP2 pixel values tensor format
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    pixel_values = torch.stack([x[0] for x in batch])
    labels       = torch.tensor([x[1] for x in batch])
    return pixel_values, labels


# DATALOADERS 
def get_dataloaders(processor):
    # Load dataset without transform — processor handles all preprocessing
    raw_data = FilteredImageFolder(
        root=DATA_DIR,
        transform=None,
        is_valid_file=lambda p: os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )

    # Stratified split: 70% train / 15% val / 15% test
    indices = list(range(len(raw_data)))
    labels  = [raw_data.targets[i] for i in indices]

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.17647,
        stratify=train_val_labels, random_state=42
    )

    print(f"Train : {len(train_idx)} | Val : {len(val_idx)} | Test : {len(test_idx)}")

    loader_args = dict(batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_siglip2,
                       num_workers=4, pin_memory=True)

    train_loader = DataLoader(
        SigLIP2Dataset(Subset(raw_data, train_idx), processor), **loader_args)
    val_loader   = DataLoader(
        SigLIP2Dataset(Subset(raw_data, val_idx),   processor), **loader_args)
    test_loader  = DataLoader(
        SigLIP2Dataset(Subset(raw_data, test_idx),  processor), **loader_args)

    return train_loader, val_loader, test_loader


# LOAD MODEL 
# device_map="auto" automatically assigns model layers to available GPU/CPU
model     = AutoModel.from_pretrained(MODEL_NAME, device_map='auto').eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"SigLIP2 So400m loaded from HuggingFace")

# LOAD DATALOADERS 
train_loader, val_loader, test_loader = get_dataloaders(processor)
