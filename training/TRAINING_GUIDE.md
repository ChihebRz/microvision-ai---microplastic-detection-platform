# Complete Training Guide: MicroVision AI

This document explains **every step** of the model training pipeline for microplastic detection.

---

## What We're Building

A **PyTorch-based object detection system** using Faster R-CNN to:
- Read your dataset images from `/data/train/` and `/data/test/`
- Parse annotations from `_annotations.csv` files
- Train a pre-trained Faster R-CNN model to detect objects in images
- Save trained weights for later inference

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Your Dataset: /data/train/ + /data/test/                  │
│  └─ Images: *.jpg files                                     │
│  └─ Labels: _annotations.csv (filename, xmin, ymin, xmax... │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   data_loader.py               │
        │ - Reads CSV                    │
        │ - Loads images                 │
        │ - Parses bboxes & labels       │
        │ Returns: (image_tensor, bboxes)│
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   PyTorch DataLoader           │
        │ - Batches samples              │
        │ - Shuffles data (training)     │
        │ - Collates via utils.collate_fn│
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   train.py (training loop)     │
        │ - Loads Faster R-CNN model     │
        │ - Forward pass: compute losses │
        │ - Backward pass: update weights│
        │ - Save checkpoints each epoch  │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   output/model_epoch_*.pth     │
        │ - Saved weights & state dict   │
        │ - Ready for inference          │
        └────────────────────────────────┘
```

---

## File Breakdown

### 1. `requirements.txt`
Lists all Python packages needed for training:
```txt
torch>=2.2.0              # Deep learning framework
torchvision>=0.17.0       # Pre-trained models (Faster R-CNN, etc.)
pandas>=2.0.0             # Reading CSV files
Pillow>=9.0.0             # Loading/processing images
tqdm                      # Progress bars
```

**Why each package?**
- **torch**: Core deep learning operations (tensors, autograd, optimizers)
- **torchvision**: Pre-trained Faster R-CNN model + image transforms
- **pandas**: Parse `_annotations.csv` → dataframe → filter by filename/bbox columns
- **Pillow**: Open `.jpg` files as PIL Images, convert to tensors
- **tqdm**: Show progress bar during epoch training

---

### 2. `data_loader.py` — Dataset Class

**Purpose**: Convert your raw image files + CSV annotations into PyTorch Dataset objects.

#### Key Functions:

**`_guess_filename_column(df)`**
- Tries to find the filename column in your CSV
- Looks for: `['filename', 'file', 'image', or first column]`
- Why? Different annotation tools use different column names

**`DetectionDataset` class**
- Inherits from `torch.utils.data.Dataset`
- Reads CSV once on initialization (in `__init__`)
- Groups annotations by filename (multiple bboxes per image)

**What happens in `__init__`:**
1. Load CSV file → pandas DataFrame
2. Detect filename column
3. Check for bbox columns: `xmin, ymin, xmax, ymax`
4. Check for class/label columns
5. Group rows by filename → list of `self.items`
   ```python
   item = {
     'filename': '100_jpg.rf.abc123.jpg',
     'boxes': [[x1, y1, x2, y2], [x1, y1, x2, y2]],  # Multiple bboxes in one image
     'labels': ['microplastic_a', 'microplastic_b']
   }
   ```

**What happens in `__getitem__(idx)`:**
1. Get item metadata (filename, boxes, labels)
2. Open image file → PIL Image → convert to RGB
3. Apply transforms (PIL → torch.Tensor via `ToTensor()`)
   - Shape: (3, H, W) — 3 channels, height, width
   - Values: [0, 1] after normalization
4. Convert bboxes to torch.Tensor, shape (N, 4) where N = number of bboxes
5. Map class labels (strings) → integers (e.g., 'microplastic' → 1, 'fiber' → 2)
6. Return tuple: `(image_tensor, target_dict)`
   ```python
   target = {
     'boxes': torch.Tensor([[x1,y1,x2,y2], ...]),  # (N, 4)
     'labels': torch.LongTensor([1, 2, ...]),       # (N,)
     'image_id': torch.tensor([idx]),
     'area': torch.Tensor([...]),                   # W*H of each bbox
     'iscrowd': torch.zeros(N)                      # For COCO eval (unused)
   }
   ```

**Smoke test (no torch required):**
```bash
python training/data_loader.py --data-dir ./data --subset train
# Output: "Loaded 577 items"  (all 577 images in train folder)
```

---

### 3. `utils.py` — Collate Function

**Purpose**: Combine multiple samples from a batch into the format Faster R-CNN expects.

```python
def collate_fn(batch):
    imgs = [b[0] for b in batch]      # List of image tensors (not stacked!)
    targets = [b[1] for b in batch]   # List of target dicts
    return imgs, targets
```

**Why not just stack images?**
- Images have varying dimensions (H, W can differ)
- Faster R-CNN handles variable-sized images via RPN (Region Proposal Network)
- We return **list of tensors**, not a single stacked tensor

**What PyTorch DataLoader does:**
```python
batch = [
  (img1_tensor, target1_dict),
  (img2_tensor, target2_dict),
  (img3_tensor, target3_dict),
]
# DataLoader calls collate_fn(batch)
imgs, targets = collate_fn(batch)
# Now: imgs = [tensor, tensor, tensor]
#      targets = [dict, dict, dict]
```

---

### 4. `train.py` — Main Training Script

**Purpose**: Load model, iterate over data, compute losses, update weights, save checkpoints.

#### Key Functions:

**`get_model(num_classes)`**
1. Load pre-trained Faster R-CNN with ResNet-50 backbone
   ```python
   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   ```
   - **What it does**: 
     - ResNet-50: extracts image features at multiple scales
     - FPN (Feature Pyramid Network): multi-scale feature maps
     - RPN (Region Proposal Network): proposes ~1000 candidate bounding boxes
     - ROI Head: refines proposals → final detections
   - **pretrained=True**: loaded from COCO pre-training (general object detection)

2. Replace head (classifier + bbox regressor)
   ```python
   in_features = model.roi_heads.box_predictor.cls_score.in_features  # 1024
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   ```
   - Old head: trained on 91 COCO classes
   - New head: trained on YOUR classes (e.g., 2 = background + microplastic)
   - This is **transfer learning**: keep ResNet backbone, retrain just the head

**`train_one_epoch(model, optimizer, dataloader, device)`**
1. Set model to training mode: `model.train()`
2. Loop over batches:
   ```python
   for images, targets in dataloader:
       # Move to GPU (if available)
       images = [img.to(device) for img in images]
       targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
       
       # Forward pass: compute losses
       loss_dict = model(images, targets)
       losses = sum(loss for loss in loss_dict.values())
       # loss_dict = {
       #   'loss_classifier': 0.5,
       #   'loss_box_reg': 0.3,
       #   'loss_objectness': 0.2,
       #   'loss_rpn_box_reg': 0.1
       # }
       
       # Backward pass: compute gradients
       optimizer.zero_grad()  # Clear old gradients
       losses.backward()      # Compute new gradients
       optimizer.step()       # Update weights
   ```
3. Return average loss for the epoch

**`main()` — Full Training Loop**
1. Parse command-line arguments:
   ```
   --data-dir: path to data folder (default: ../data)
   --subset: train or test (default: train)
   --epochs: number of passes over dataset (default: 5)
   --batch-size: samples per batch (default: 4)
   --lr: learning rate (default: 0.005)
   --output-dir: where to save checkpoints (default: ./output)
   ```

2. Load dataset:
   ```python
   dataset = DetectionDataset(images_dir, csv_path)
   dataloader = DataLoader(
       dataset, 
       batch_size=args.batch_size, 
       shuffle=True,  # Random order each epoch
       collate_fn=collate_fn
   )
   ```

3. Load model and set to device (GPU or CPU):
   ```python
   model = get_model(num_classes=2)  # background + microplastic
   model.to(device)
   ```

4. Create optimizer (SGD with momentum):
   ```python
   optimizer = torch.optim.SGD(
       [p for p in model.parameters() if p.requires_grad],
       lr=0.005,
       momentum=0.9,
       weight_decay=0.0005  # L2 regularization
   )
   ```
   - **SGD**: Stochastic Gradient Descent — update after each batch
   - **momentum=0.9**: Accelerate convergence, smooth noisy gradients
   - **weight_decay**: Penalty for large weights (prevents overfitting)

5. Loop over epochs:
   ```python
   for epoch in range(1, args.epochs + 1):
       loss = train_one_epoch(model, optimizer, dataloader, device)
       print(f'Epoch {epoch} - loss: {loss:.4f}')
       
       # Save checkpoint
       torch.save(model.state_dict(), f'./output/model_epoch_{epoch}.pth')
   ```

---

## Quick Start Commands

### 1. Set up Python Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r training/requirements.txt
```

### 2. Smoke Test (Verify CSV + images load)
```bash
python training/data_loader.py --data-dir ./data --subset train
# Output: "Loaded 577 items"
```

### 3. Run Training (1 epoch, small batch — fast!)
```bash
python training/train.py \
  --data-dir ./data \
  --subset train \
  --epochs 1 \
  --batch-size 4 \
  --output-dir ./training/output
```

### 4. Run Full Training (5 epochs, batch-size 4)
```bash
python training/train.py \
  --data-dir ./data \
  --subset train \
  --epochs 5 \
  --batch-size 4 \
  --lr 0.005 \
  --output-dir ./training/output
```

### 5. Check Saved Weights
```bash
ls -lh training/output/
# Shows: model_epoch_1.pth, model_epoch_2.pth, ...
# Each file ~165 MB (ResNet-50 weights)
```

---

## Understanding Training Output

When you run training, you'll see:
```
Epoch 1/5 - loss: 2.3456 - time: 123.5s
Saved ./training/output/model_epoch_1.pth
Epoch 2/5 - loss: 1.8234 - time: 120.3s
Saved ./training/output/model_epoch_2.pth
...
```

**Line breakdown:**
- **Epoch 1/5**: Currently on epoch 1 of 5 total
- **loss: 2.3456**: Average loss across all batches this epoch
  - Starts high (~2-3), decreases over epochs
  - Lower = model getting better at detecting objects
- **time: 123.5s**: How long this epoch took (depends on batch-size, dataset size, CPU/GPU)
- **model_epoch_1.pth**: Checkpoint saved (contains all weights)

**Loss components** (in `train.py` they're summed):
- `loss_classifier`: Is this region an object? (binary classification)
- `loss_box_reg`: Bounding box coordinates (regression)
- `loss_objectness`: Does RPN think this is an object?
- `loss_rpn_box_reg`: RPN bounding box refinement

---

## How to Use the Trained Model for Inference

After training, you can load the checkpoint and make predictions:

```python
import torch
from train import get_model

# Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load('./training/output/model_epoch_5.pth'))
model.eval()  # Set to evaluation mode

# Inference on a single image
from PIL import Image
import torchvision.transforms as T

img = Image.open('my_image.jpg').convert('RGB')
img_tensor = T.ToTensor()(img).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    outputs = model([img_tensor])

# outputs[0] = {
#   'boxes': tensor([[x1, y1, x2, y2], ...]),  # Detected boxes
#   'labels': tensor([1, 1, ...]),              # Class IDs
#   'scores': tensor([0.95, 0.87, ...])         # Confidence scores
# }
```

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Fix**: Make sure venv is activated:
```bash
source .venv/bin/activate
```

### Issue: "FileNotFoundError: /path/to/_annotations.csv"
**Fix**: Check that CSV exists:
```bash
ls -la data/train/_annotations.csv
ls -la data/test/_annotations.csv
```

### Issue: "Could not find a filename column in the CSV"
**Fix**: Check CSV header — must have one of: `filename`, `file`, or `image`
```bash
head data/train/_annotations.csv
```

### Issue: Training is very slow (CPU only)
**Fix**: GPU training is much faster. If you have CUDA:
```bash
# Install GPU version instead
pip uninstall torch -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory (GPU)
**Fix**: Reduce batch size:
```bash
python training/train.py --batch-size 2
```

---

## Summary

| Step | File | What It Does |
|------|------|--------------|
| 1 | `requirements.txt` | List Python packages to install |
| 2 | `data_loader.py` | Read CSV + images → PyTorch Dataset |
| 3 | `utils.py` | Batch samples for DataLoader |
| 4 | `train.py` | Main training loop: load model, iterate, optimize |
| 5 | `output/model_epoch_*.pth` | Saved weights after each epoch |

**Flow**:
```
CSV + Images → DetectionDataset → DataLoader → train.py → Faster R-CNN → Loss → Backprop → Updated Weights → output/
```

---

## Next Steps

1. ✅ Smoke-test data loader (verify data loads)
2. ⏳ Run full training (1-5 epochs)
3. ⏳ Evaluate on test set (precision, recall, mAP)
4. ⏳ Integrate trained model into React frontend `/services/geminiService.ts`

---

**Questions?** Check the code comments in each file or re-read the relevant section above.
