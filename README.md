# SigLIP2 So400m Loader

Loads SigLIP2 So400m from HuggingFace for zero-shot embedding extraction. No training required.

## Model

SigLIP2 So400m with 400M parameters pretrained on WebLI (10 billion image-text pairs across 109 languages) using a combination of four training objectives: sigmoid contrastive loss, captioning-based pretraining, self-distillation, and masked prediction. Embeddings of 1152 dimensions are extracted from the vision model pooler output without any fine-tuning. Input images are processed at 384×384 resolution.

## Usage

The script outputs `model`, `train_loader`, `val_loader` and `test_loader` ready to import into `extractor.py`. SigLIP2 requires its own HuggingFace processor for preprocessing — standard torchvision transforms are not compatible.

```python
from siglip2_loader import model, train_loader, val_loader, test_loader
```

Batch size is set to 8 due to the large model size. Model weights are downloaded to the local drive by default:

```python
os.environ['HF_HOME'] = '/your/path/here'
```

## Key Details

| Property | Value |
|----------|-------|
| Architecture | ViT-So400m/14 |
| Pretrained on | WebLI |
| Parameters | 400M |
| Embedding dim | 1152 |
| Input size | 384×384 |
| Batch size | 8 |
| Fine-tuning | None (zero-shot) |
