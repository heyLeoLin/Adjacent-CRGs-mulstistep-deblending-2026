import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, flag=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.flag = flag
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.flag == 0:
                return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
            else:
                return [base_lr * 0.5 * (self.last_epoch / self.warmup_epochs + 1) for base_lr in self.base_lrs]
        else:
            return [base_lr * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))) / 2 for base_lr in self.base_lrs]

# Example
#   optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
#   scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=30, total_epochs=500)
#   after training
#   scheduler.step()