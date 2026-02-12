import os
from typing import List, Dict, Tuple, Optional
from PIL import Image

# avoid importing torch at module-import time so a light smoke-test can run
try:
    import torch
except Exception:
    torch = None

try:
    from torch.utils.data import Dataset
except Exception:
    class Dataset:  # fallback placeholder for smoke-tests when torch is not installed
        pass

import pandas as pd


def _guess_filename_column(df: pd.DataFrame) -> Optional[str]:
    for c in ['filename', 'file', 'image', df.columns[0]]:
        if c in df.columns:
            return c
    return None


class DetectionDataset(Dataset):
    """A minimal dataset for object detection.

    Supports CSVs with columns for filename and optionally xmin,ymin,xmax,ymax and class/label.
    """

    def __init__(self, images_dir: str, annotations_csv: str, transforms=None):
        self.images_dir = images_dir
        self.annotations_csv = annotations_csv
        if transforms is not None:
            self.transforms = transforms
        else:
            try:
                import torchvision.transforms as T
                self.transforms = T.ToTensor()
            except Exception:
                # fallback: return PIL image as-is for smoke tests
                self.transforms = lambda img: img

        if not os.path.exists(self.annotations_csv):
            raise FileNotFoundError(self.annotations_csv)

        df = pd.read_csv(self.annotations_csv)
        fname_col = _guess_filename_column(df)
        if fname_col is None:
            raise ValueError('Could not find a filename column in the CSV')

        df = df.rename(columns={fname_col: 'filename'})

        # normalize possible bbox column names
        col_map = {c.lower(): c for c in df.columns}
        self.has_bbox = all(k in col_map for k in ['xmin', 'ymin', 'xmax', 'ymax'])
        self.has_label = any(k in col_map for k in ['class', 'label'])

        # group rows by filename
        grouped = df.groupby('filename')
        self.items: List[Dict] = []

        for fname, group in grouped:
            entry = {'filename': fname}
            if self.has_bbox:
                boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float)
                entry['boxes'] = boxes
            else:
                entry['boxes'] = []

            if self.has_label:
                label_col = 'class' if 'class' in col_map else ('label' if 'label' in col_map else None)
                if label_col:
                    labels = group[label_col].values.tolist()
                    entry['labels'] = labels
                else:
                    entry['labels'] = []
            else:
                entry['labels'] = []

            self.items.append(entry)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = os.path.join(self.images_dir, item['filename'])
        if not os.path.exists(path):
            # try some common alternate names (strip quotes)
            alt = item['filename'].strip('"')
            path = os.path.join(self.images_dir, alt)
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        img = Image.open(path).convert('RGB')
        img_tensor = self.transforms(img)

        boxes = []
        labels = []
        if len(item.get('boxes', [])):
            # convert to torch tensors when torch is available, otherwise keep as lists
            if torch is not None:
                boxes = torch.as_tensor(item['boxes'], dtype=torch.float32)
                # if labels present try to map to integers else use 1
                if len(item.get('labels', [])):
                    lbls = item['labels']
                    mapping = {v: i + 1 for i, v in enumerate(sorted(set(lbls)))}
                    labels = torch.as_tensor([mapping.get(v, 1) for v in lbls], dtype=torch.int64)
                else:
                    labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            else:
                boxes = item['boxes']
                labels = item.get('labels', []) or []
        else:
            if torch is not None:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = []
                labels = []

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        if torch is not None:
            target['image_id'] = torch.tensor([idx])
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() else torch.tensor([])
            target['iscrowd'] = torch.zeros((labels.shape[0],), dtype=torch.int64)
        else:
            target['image_id'] = [idx]
            target['area'] = []
            target['iscrowd'] = []

        return img_tensor, target


if __name__ == '__main__':
    # quick smoke test when run directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--subset', default='train')
    args = parser.parse_args()
    csv = os.path.join(args.data_dir, args.subset, '_annotations.csv')
    imgs = os.path.join(args.data_dir, args.subset)
    ds = DetectionDataset(imgs, csv)
    print('Loaded', len(ds), 'items')
