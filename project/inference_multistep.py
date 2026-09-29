'''
Author: Lin Shicong
Task: PDB Data Deblending (3CRG)
Description: 
- Architecture: lightweight U-Net (Dropout 0.5)
- Optimizer: AdamW with Warmup Cosine Annealing Learning Rate
- Inference
'''

import os
import argparse
import numpy as np
import scipy.io as sio
import hdf5storage
import torch
from torch.utils.data import DataLoader, TensorDataset

# Local module imports (ensure these are included in your open-source repo)
from unet_tool import Unet, snr_fn2d

def load_mat(path, key):
    data = hdf5storage.loadmat(path)
    if key not in data:
        raise KeyError(f"{key} not found in {path}. Available keys: {list(data.keys())}")
    return data[key]

def predict(model, x, device, batch_size=2):
    x_tensor = torch.from_numpy(x.astype(np.float32))
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)
    pred = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            yb = model(xb.to(device))
            pred.append(yb.cpu().numpy())
    pred = np.concatenate(pred, axis=0)
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True,
                        help="Step 1: dataset_3channel.mat; Step 2: data_pred_bnss.mat")
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="hun_test")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Both Step 1 and Step 2 use the same 3-channel -> 1-channel U-Net.
    model = Unet(in_channel=3, out_channel=1).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {args.model}")

    x = load_mat(args.input, args.key)
    print(f"Input: {args.input}")
    print(f"Input key: {args.key}")
    print(f"Input shape: {x.shape}")

    if x.ndim != 4:
        raise ValueError(f"Expected input shape [N, 3, nt, nx], got {x.shape}")
    if x.shape[1] != 3:
        raise ValueError(
            f"Model expects 3 input channels, but got {x.shape[1]}. "
            "For Step 2, use the BNSS-updated 3-channel dataset."
        )

    pred = predict(model, x, device, args.batch_size)
    print(f"Prediction shape: {pred.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sio.savemat(args.output, {"pred_test": pred})
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()



# ==============================================================================
# Execution Examples
# ==============================================================================
# # Step 1
# python inference_multistep.py \
#     --step 1 \
#     --model ../result/3c1_step1/model.pth \
#     --input ../data/dataset_3channel.mat \
#     --output ../result/3c1_step1/data_pred.mat

# # Step 2
# python inference_multistep.py \
#     --step 2 \
#     --model ../result/3c1_step2/model.pth \
#     --input ../result/3c1_step1/data_pred_bnss.mat \
#     --output ../result/3c1_step2/data_pred.mat