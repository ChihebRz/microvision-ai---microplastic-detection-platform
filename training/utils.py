from typing import List, Tuple
import torch

def collate_fn(batch: List[Tuple[torch.Tensor, dict]]):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return imgs, targets
