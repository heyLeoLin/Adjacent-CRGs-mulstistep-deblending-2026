
'''
Author: Lin Shicong
Task: PDB Data Deblending (3CRG)
Description: 
- Architecture: lightweight U-Net (Dropout 0.5)
- Optimizer: AdamW with Warmup Cosine Annealing Learning Rate
- Training: Supervised Learning
'''

import os
import time
import argparse
import numpy as np
import scipy.io as sio
import hdf5storage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from unet_tool import Unet, snr_fn2d
from warmup_tool import WarmupCosineAnnealingLR

# Default training configurations and paths mapped by stage index
STEP_CONFIGS = {
    1: {
        "epochs": 150,
        "lr": 4e-3,
        "flag": 0,
        "input_x": "../data/dataset_3channel.mat",
        "input_y": "../data/dataset_3channel.mat",
        "pretrained": None,
        "save_dir": "../result/3c1_step1",
    },
    2: {
        "epochs": 100,
        "lr": 1e-3,
        "flag": 1,
        "input_x": "../result/3c1_step1/data_pred_bnss.mat",
        "input_y": "../data/dataset_1channel.mat",
        "pretrained": "../result/3c1_step1/model.pth",
        "save_dir": "../result/3c1_step2",
    },
    3: {
        "epochs": 50,
        "lr": 5e-4,
        "flag": 1,
        "input_x": "../result/3c1_step2/data_pred_bnss.mat",
        "input_y": "../data/dataset_1channel.mat",
        "pretrained": "../result/3c1_step2/model.pth",
        "save_dir": "../result/3c1_step3",
    },
}

def set_seed(seed=0):
    # Ensure reproducible outcomes across random and CUDA backends
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_mat(path, key):
    # Safely load targeted key from MAT file with diagnostic feedback
    data = hdf5storage.loadmat(path)
    if key not in data:
        raise KeyError(f"{key} not found in {path}. Available keys: {list(data.keys())}")
    return data[key]

def build_dataloader(x, y, batch_size=2, shuffle=False):
    # Validate input dimensions and enforce 4D tensor formatting
    if x.ndim != 4:
        raise ValueError(f"Expected input x shape [N, 3, nt, nx], got {x.shape}")
    if x.shape[1] != 3:
        raise ValueError(f"Model expects 3 input channels, got {x.shape[1]}")
    
    # Expand channel dimension for target if loaded as 3D array
    if y.ndim == 3:
        y = np.expand_dims(y, axis=1)
    if y.ndim != 4 or y.shape[1] != 1:
        raise ValueError(f"Expected target y shape [N, 1, nt, nx], got {y.shape}")

    x_tensor = torch.from_numpy(x.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, loader, optimizer, loss_fn, device, crop_nt=None):
    # Execute one full forward and backward training epoch
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        if crop_nt is not None:
            y_hat = y_hat[:, :, :crop_nt, :]
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, loss_fn, device, crop_nt=None):
    # Compute average validation loss and SNR over the entire dataloader
    model.eval()
    total_loss, total_snr, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            if crop_nt is not None:
                y_hat = y_hat[:, :, :crop_nt, :]
            batch_sz = y.shape[0]
            loss = loss_fn(y_hat, y)
            total_loss += loss.item() * batch_sz
            total_snr += snr_fn2d(y_hat, y) * batch_sz
            total_samples += batch_sz
    return total_loss / total_samples, total_snr / total_samples

def predict(model, loader, device, crop_nt=None):
    # Perform batched inference and squeeze redundant channel dimensions
    model.eval()
    pred = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = model(x)
            if crop_nt is not None:
                y_hat = y_hat[:, :, :crop_nt, :]
            y_np = y_hat.cpu().numpy()
            if y_np.ndim == 4 and y_np.shape[1] == 1:
                y_np = y_np[:, 0]
            pred.append(y_np)
    return np.concatenate(pred, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Multi-stage Training for PDB Data Deblending")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], required=True,
                        help="Select training stage: 1 (supervised), 2 (refinement 1), or 3 (refinement 2)")
    parser.add_argument("--epochs", type=int, default=None, help="Custom epochs count to override step default")
    parser.add_argument("--lr", type=float, default=None, help="Custom learning rate to override step default")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_x", default=None, help="Custom file path for input x")
    parser.add_argument("--input_y", default=None, help="Custom file path for ground truth y")
    parser.add_argument("--pretrained", default=None, help="Custom file path to pre-trained checkpoint")
    parser.add_argument("--save_dir", default=None, help="Destination directory for outputs")
    args = parser.parse_args()

    # Resolve stage configurations with optional command line overrides
    cfg = STEP_CONFIGS[args.step]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    initial_lr = args.lr if args.lr is not None else cfg["lr"]
    input_x_path = args.input_x or cfg["input_x"]
    input_y_path = args.input_y or cfg["input_y"]
    pretrained_path = args.pretrained or cfg["pretrained"]
    save_dir = args.save_dir or cfg["save_dir"]

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running Step {args.step} | Epochs: {epochs} | Batch size: {args.batch_size} | LR: {initial_lr}")

    # Load dataset partitions and verify shapes
    x_train = load_mat(input_x_path, "hun_train")
    x_valid = load_mat(input_x_path, "hun_valid")
    x_test = load_mat(input_x_path, "hun_test")

    y_train = load_mat(input_y_path, "data_train")
    y_valid = load_mat(input_y_path, "data_valid")
    y_test = load_mat(input_y_path, "data_test")

    nt = x_train.shape[2]
    print(f"Input X path: {input_x_path} | Shape: {x_train.shape}")
    print(f"Target Y path: {input_y_path} | Shape: {y_train.shape}")

    train_loader = build_dataloader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = build_dataloader(x_valid, y_valid, batch_size=args.batch_size, shuffle=False)
    test_loader = build_dataloader(x_test, y_test, batch_size=args.batch_size, shuffle=False)

    # Initialize model architecture and restore pre-trained state when specified
    model = Unet(in_channel=3, out_channel=1).to(device)
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Checkpoint not found at {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pre-trained model: {pretrained_path}")

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.95))
    scheduler = WarmupCosineAnnealingLR(
        optimizer, warmup_epochs=epochs / 10, total_epochs=epochs, flag=cfg["flag"]
    )
    loss_fn = nn.MSELoss()

    total_train_loss, total_valid_loss = [], []
    total_train_snr, total_valid_snr = [], []
    all_lr = []

    start_time_global = time.time()
    print("Starting training process...")

    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_epoch(model, train_loader, optimizer, loss_fn, device, crop_nt=nt)
        scheduler.step()

        loss_tr, snr_tr = evaluate(model, train_loader, loss_fn, device, crop_nt=nt)
        loss_val, snr_val = evaluate(model, valid_loader, loss_fn, device, crop_nt=nt)

        total_train_loss.append(loss_tr)
        total_train_snr.append(snr_tr)
        total_valid_loss.append(loss_val)
        total_valid_snr.append(snr_val)
        all_lr.append(current_lr)

        run_time = time.time() - epoch_start
        elapsed_min = (time.time() - start_time_global) / 60.0
        eta_min = run_time * (epochs - 1 - epoch) / 60.0

        print(
            f"Epoch: {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss: {loss_tr:.5f} | Train SNR: {snr_tr:.3f} | "
            f"Val Loss: {loss_val:.5f} | Val SNR: {snr_val:.3f} | "
            f"LR: {current_lr:.7f} | Elapsed: {elapsed_min:.1f}m | ETA: {eta_min:.1f}m",
            flush=True,
        )

    # Persist model weights and training curves
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint: {checkpoint_path}")

    metrics_path = os.path.join(save_dir, "data_para.mat")
    sio.savemat(
        metrics_path,
        {
            "loss_train": total_train_loss,
            "loss_test": total_valid_loss,
            "snr_train": total_train_snr,
            "snr_test": total_valid_snr,
            "lr": all_lr,
        },
    )
    print(f"Saved metric curves: {metrics_path}")

    # Generate and save final test predictions
    eval_train_loader = build_dataloader(x_train, y_train, batch_size=args.batch_size, shuffle=False)
    _, final_train_snr = evaluate(model, eval_train_loader, loss_fn, device, crop_nt=nt)
    _, final_test_snr = evaluate(model, test_loader, loss_fn, device, crop_nt=nt)
    print(f"Final Train SNR: {final_train_snr:.3f} | Final Test SNR: {final_test_snr:.3f}")

    pred_test = predict(model, test_loader, device, crop_nt=nt)
    print(f"Prediction output shape: {pred_test.shape}")

    pred_save_path = os.path.join(save_dir, "data_pred.mat")
    sio.savemat(pred_save_path, {"pred_test": pred_test})
    print(f"Saved predictions: {pred_save_path}")

if __name__ == "__main__":
    main()


# ==============================================================================
# Execution Examples
# ==============================================================================
# # Step 1: Initial Supervised Stage
# python train_multistep.py --step 1
#
# # Step 2: First Transfer Refinement Stage
# python train_multistep.py --step 2
#
# # Step 3: Second Transfer Refinement Stage
# python train_multistep.py --step 3
#
# # Example with custom hyperparameters:
# python train_multistep.py --step 2 --epochs 80 --lr 8e-4 --batch_size 4
