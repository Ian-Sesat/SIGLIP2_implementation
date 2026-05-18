"""
SigLIP2 Zero-Shot Evaluation
=============================
Zero-shot evaluation of SigLIP2 So400m on custom dataset.

No training required — uses pretrained weights directly.

Evaluates on:
- Image Retrieval : mAP, Recall@1, Recall@5
- Classification  : kNN@21
"""

import os
import numpy as np
import warnings
import faiss

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.cuda.amp import autocast

from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────
DATA_DIR  = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'
SAVE_DIR  = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'

EMBEDDINGS = os.path.join(SAVE_DIR, 'embeddings_siglip2_zeroshot.npz')

# Model
MODEL_NAME    = 'google/siglip2-so400m-patch14-384'
EMBEDDING_DIM = 1152

# Dataset
NUM_CLASSES    = 100
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'}

# Extraction
EXTRACTION_BATCH_SIZE = 8  # small for 400M model safety

# Run flags
RUN_EXTRACTION = True
RUN_EVALUATION = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


# 1. DATASET 
class FilteredImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [
            folder for folder in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, folder))
            and folder not in IGNORE_FOLDERS
        ]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx


class SigLIP2Dataset(torch.utils.data.Dataset):
    """
    Custom dataset that uses SigLIP2 processor for preprocessing.
    Returns processed tensors ready for SigLIP2.
    """
    def __init__(self, subset, processor):
        self.subset    = subset
        self.processor = processor

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            image, label = self.subset[idx]
            # image is already a PIL image from ImageFolder
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            # squeeze batch dimension
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, label
        except Exception:
            return None


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    pixel_values = torch.stack([x[0] for x in batch])
    labels       = torch.tensor([x[1] for x in batch])
    return pixel_values, labels


def get_dataloaders(processor, batch_size):
    # Load dataset without transform (SigLIP2Dataset handles preprocessing)
    raw_data = FilteredImageFolder(
        root=DATA_DIR,
        transform=None,  # no transform — processor handles it
        is_valid_file=lambda path: os.path.splitext(path)[1].lower() in IMAGE_EXTS
    )

    indices = list(range(len(raw_data)))
    labels  = [raw_data.targets[i] for i in indices]

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.17647, stratify=train_val_labels, random_state=42
    )

    train_subset = Subset(raw_data, train_idx)
    val_subset   = Subset(raw_data, val_idx)
    test_subset  = Subset(raw_data, test_idx)

    train_dataset = SigLIP2Dataset(train_subset, processor)
    val_dataset   = SigLIP2Dataset(val_subset,   processor)
    test_dataset  = SigLIP2Dataset(test_subset,  processor)

    print(f"Train : {len(train_dataset)} | "
          f"Val : {len(val_dataset)} | "
          f"Test : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, raw_data.classes


# 2. EMBEDDING EXTRACTION 
def extract_embeddings(loader, model, device):
    """Extract image embeddings using SigLIP2 image encoder."""
    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if batch is None:
                continue
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)

            with autocast():
                outputs    = model.vision_model(pixel_values=pixel_values)
                embeddings = outputs.pooler_output  # ← get the actual tensor
                embeddings = nn.functional.normalize(embeddings, dim=-1)

            all_embeddings.append(embeddings.cpu().float().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_labels,     axis=0)
    )


# 3. RETRIEVAL EVALUATION 
def build_faiss_index(embeddings):
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors ")
    return index


def evaluate_map(test_embeddings, test_labels, index, train_labels):
    """Calculate mean Average Precision."""
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    aps        = []
    chunk_size = 1000

    for start in tqdm(range(0, len(query), chunk_size), desc="mAP Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, index.ntotal)

        for i, idx in enumerate(indices):
            query_label    = test_labels[start + i]
            retrieved      = train_labels[idx]
            total_relevant = (train_labels == query_label).sum()

            ap      = 0.0
            correct = 0

            for rank, label in enumerate(retrieved, 1):
                if label == query_label:
                    correct += 1
                    ap      += correct / rank

            ap = ap / total_relevant if total_relevant > 0 else 0.0
            aps.append(ap)

    map_score = np.mean(aps)
    print(f"  mAP : {map_score:.4f}")
    return map_score


def evaluate_recall(test_embeddings, test_labels, index, train_labels, k=1):
    """Calculate Recall@K."""
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    correct    = 0
    chunk_size = 10000

    for start in tqdm(range(0, len(query), chunk_size), desc=f"Recall@{k} Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, k)

        for i, idx in enumerate(indices):
            query_label  = test_labels[start + i]
            top_k_labels = train_labels[idx]
            if query_label in top_k_labels:
                correct += 1

    recall = correct / len(test_embeddings)
    print(f"  Recall@{k} : {recall:.4f}")
    return recall


def knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21):
    """Calculate kNN classification accuracy."""
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    correct    = 0
    chunk_size = 10000

    for start in tqdm(range(0, len(query), chunk_size), desc="kNN Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, k)

        for i, idx in enumerate(indices):
            query_label     = test_labels[start + i]
            top_k_labels    = train_labels[idx]
            votes           = np.bincount(top_k_labels, minlength=NUM_CLASSES)
            predicted_class = np.argmax(votes)
            if predicted_class == query_label:
                correct += 1

    accuracy = correct / len(test_embeddings)
    print(f"  kNN Accuracy (k={k}) : {accuracy:.4f}")
    return accuracy


# MAIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device  : {device}")
print(f"Model         : {MODEL_NAME}")
print(f"Embedding dim : {EMBEDDING_DIM}")

# Load SigLIP2
print("\nLoading SigLIP2 So400m ...")
model     = AutoModel.from_pretrained(
                MODEL_NAME,
                device_map="auto"
            ).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("SigLIP2 loaded! ")

# Get dataloaders
train_loader, val_loader, test_loader, classes = get_dataloaders(
    processor, EXTRACTION_BATCH_SIZE
)
print(f"Classes : {len(classes)}")


# EMBEDDING EXTRACTION

if RUN_EXTRACTION:
    print("\n" + "="*60)
    print("EXTRACTION — SigLIP2 So400m")
    print("="*60)

    torch.cuda.empty_cache()

    print("Extracting train embeddings ...")
    train_embeddings, train_labels = extract_embeddings(train_loader, model, device)
    print("Extracting val embeddings ...")
    val_embeddings,   val_labels   = extract_embeddings(val_loader,   model, device)
    print("Extracting test embeddings ...")
    test_embeddings,  test_labels  = extract_embeddings(test_loader,  model, device)

    print(f"Train embeddings : {train_embeddings.shape}")
    print(f"Val embeddings   : {val_embeddings.shape}")
    print(f"Test embeddings  : {test_embeddings.shape}")

    np.savez(EMBEDDINGS,
             train_embeddings=train_embeddings, train_labels=train_labels,
             val_embeddings=val_embeddings,     val_labels=val_labels,
             test_embeddings=test_embeddings,   test_labels=test_labels)
    print(f"Embeddings saved → {EMBEDDINGS}")

else:
    data             = np.load(EMBEDDINGS)
    train_embeddings = data['train_embeddings']
    train_labels     = data['train_labels']
    val_embeddings   = data['val_embeddings']
    val_labels       = data['val_labels']
    test_embeddings  = data['test_embeddings']
    test_labels      = data['test_labels']
    print("Embeddings loaded!")


# EVALUATION

if RUN_EVALUATION:
    print("\n" + "="*60)
    print("EVALUATION — SigLIP2 So400m Zero-Shot")
    print("="*60)

    torch.cuda.empty_cache()
    index = build_faiss_index(train_embeddings)

    print("\n── Image Retrieval ──")
    map_score = evaluate_map(test_embeddings, test_labels, index, train_labels)
    recall_1  = evaluate_recall(test_embeddings, test_labels, index, train_labels, k=1)
    recall_5  = evaluate_recall(test_embeddings, test_labels, index, train_labels, k=5)

    print("\n── kNN Classification ──")
    knn_acc = knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21)

    print("\n" + "="*60)
    print("FINAL RESULTS — SigLIP2 So400m Zero-Shot")
    print("="*60)
    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 35)
    print(f"{'mAP':<25} {map_score*100:>9.2f}%")
    print(f"{'Recall@1':<25} {recall_1*100:>9.2f}%")
    print(f"{'Recall@5':<25} {recall_5*100:>9.2f}%")
    print(f"{'kNN Accuracy @21':<25} {knn_acc*100:>9.2f}%")
    print("-" * 35)
    print("\n ALL DONE! ")