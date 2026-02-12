# Training (PyTorch)

This folder contains a minimal PyTorch training scaffold for object detection (Faster R-CNN).

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

2. Run training (example):

```bash
python training/train.py --data-dir ./data --subset train --epochs 5 --output-dir ./training/output
```

What the code does

- `data_loader.py`: a `torch.utils.data.Dataset` that reads `_annotations.csv` and images under `data/<subset>/`.
- `train.py`: simple training loop using `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- `utils.py`: helper utilities (collate function).

Notes

- The `_annotations.csv` files should include at least a filename column. If bounding boxes and class labels are present, the loader will use them. Supported column names: `filename`, `file`, `image`, `xmin`, `ymin`, `xmax`, `ymax`, `class`, `label`.
- This scaffold is minimal and intended for experimentation. For heavy training and production, use proper data pipelines, augmentation, and distributed training.
