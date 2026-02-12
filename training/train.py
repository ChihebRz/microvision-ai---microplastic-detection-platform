import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_loader import DetectionDataset
from utils import collate_fn


def get_model(num_classes: int):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(dataloader, desc='train'):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(dataloader)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../data', help='path to data folder')
    p.add_argument('--subset', default='train', help='train subset folder name (train/test)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--output-dir', default='./output')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_dir = os.path.join(args.data_dir, args.subset)
    csv_path = os.path.join(args.data_dir, args.subset, '_annotations.csv')

    dataset = DetectionDataset(images_dir, csv_path)

    # try to infer number of classes from dataset labels (best-effort)
    num_classes = 2  # background + at least one
    # attempt to infer from dataset items
    try:
        labels = set()
        for it in dataset.items:
            for l in it.get('labels', []):
                labels.add(str(l))
        if labels:
            num_classes = len(labels) + 1
    except Exception:
        pass

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, optimizer, dataloader, device)
        elapsed = time.time() - start
        print(f'Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - time: {elapsed:.1f}s')

        # save checkpoint each epoch
        ckpt = os.path.join(args.output_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print('Saved', ckpt)


if __name__ == '__main__':
    main()
