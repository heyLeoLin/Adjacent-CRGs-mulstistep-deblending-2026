'''
Author: Lin Shicong
Task: PDB Data Deblending - BNSS Operation
Description: 
- Performs a Blending Noise Simulation-Subtraction (BNSS) operation.
- Uses results from the previous-step deblending to estimate blending noise.
- Constructs and updates the dataset (supports both 3-CRG and 1-CRG formats).
'''

import os
import argparse
import numpy as np
import scipy.io as sio
import hdf5storage

# Default configuration parameters mapped by refinement step
STEP_CONFIGS = {
    1: {
        "pred_path": "../result/3c1_step1/data_pred.mat",
        "output_path": "../result/3c1_step1/data_pred_bnss.mat",
    },
    2: {
        "pred_path": "../result/3c1_step2/data_pred.mat",
        "output_path": "../result/3c1_step2/data_pred_bnss.mat",
    },
}

def load_mat(path, key):
    # Load targeted array from MAT file with diagnostic key validation
    data = hdf5storage.loadmat(path)
    if key not in data:
        raise KeyError(f"{key} not found in {path}. Available keys: {list(data.keys())}")
    return data[key]

def simulate_blending(pred_test, shooting_indices, nt, nx):
    # Simulate blended wavefield by accumulating overlapping gathers across active shooting positions
    nr = pred_test.shape[0]
    pred_blended = np.zeros_like(pred_test)
    continuous_record = np.zeros((2 * nx * nt,))
    gather_matrix = np.zeros((nt, nx))

    for j in range(nr):
        gather = pred_test[j, :, :]
        continuous_record.fill(0)
        gather_matrix.fill(0)

        for i, pos in enumerate(shooting_indices):
            continuous_record[pos - 1 : pos + nt - 1] += gather[:, i]

        for idx, pos in enumerate(shooting_indices):
            gather_matrix[:, idx] = continuous_record[pos - 1 : pos + nt - 1]

        pred_blended[j, :, :] = gather_matrix
    return pred_blended

def build_3crg_dataset(pred_bnss, label_data, train_idx, valid_idx, test_idx, nr, nt, nx):
    # Assemble 3-channel input arrays incorporating neighboring gather context and boundary padding
    hun_all = np.expand_dims(pred_bnss, axis=1)

    def extract_subset(indices):
        subset_hun = np.zeros((len(indices), 3, nt, nx))
        subset_data = np.zeros((len(indices), nt, nx))
        for i, j in enumerate(indices):
            if j == 1:
                subset_hun[i, 0, :, :] = hun_all[j - 1, 0, :, :]
                subset_hun[i, 1, :, :] = hun_all[j - 1, 0, :, :]
                subset_hun[i, 2, :, :] = hun_all[j, 0, :, :]
            elif j == nr:
                subset_hun[i, 0, :, :] = hun_all[j - 2, 0, :, :]
                subset_hun[i, 1, :, :] = hun_all[j - 1, 0, :, :]
                subset_hun[i, 2, :, :] = hun_all[j - 1, 0, :, :]
            else:
                subset_hun[i, 0, :, :] = hun_all[j - 2, 0, :, :]
                subset_hun[i, 1, :, :] = hun_all[j - 1, 0, :, :]
                subset_hun[i, 2, :, :] = hun_all[j, 0, :, :]
            subset_data[i, :, :] = label_data[j - 1, :, :]
        return subset_hun, subset_data

    hun_train, _ = extract_subset(train_idx)
    hun_valid, _ = extract_subset(valid_idx)
    hun_test, _ = extract_subset(test_idx)
    return hun_train, hun_valid, hun_test

def build_1crg_dataset(pred_bnss, train_idx, valid_idx, test_idx):
    # Slice single-channel gathers directly according to split indices
    hun_train = pred_bnss[train_idx - 1, :, :]
    hun_valid = pred_bnss[valid_idx - 1, :, :]
    hun_test = pred_bnss[test_idx - 1, :, :]
    return hun_train, hun_valid, hun_test

def main():
    parser = argparse.ArgumentParser(description="Blending Noise Simulation-Subtraction (BNSS) Data Builder")
    parser.add_argument("--step", type=int, choices=[1, 2], required=True,
                        help="Select previous step whose prediction will be processed: 1 or 2")
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3,
                        help="Dataset format channel dimension: 3 for 3-CRG or 1 for 1-CRG")
    parser.add_argument("--raw_data", default="../data/dataset_1channel.mat",
                        help="Path to synthetic dataset MAT containing target labels and shooting schedules")
    parser.add_argument("--pred_input", default=None, help="Custom path to predicted MAT file")
    parser.add_argument("--output", default=None, help="Custom output path for generated BNSS MAT file")
    args = parser.parse_args()

    cfg = STEP_CONFIGS[args.step]
    pred_path = args.pred_input or cfg["pred_path"]
    output_path = args.output or cfg["output_path"]

    # Verify input prediction availability and retrieve raw geometry data
    pred_test = load_mat(pred_path, "pred_test")
    pdb_data = load_mat(args.raw_data, "hun_test")
    label_data = load_mat(args.raw_data, "data_test")
    shooting_schedule = load_mat(args.raw_data, "Data_Shooting_he").astype(np.int32).flatten()

    nr, nt, nx = pred_test.shape
    print(f"Loaded prediction: {pred_path} with shape {pred_test.shape}")
    print(f"Loaded reference dataset: {args.raw_data}")

    # Isolate valid shooting records and simulate blending noise
    valid_shooting = shooting_schedule[shooting_schedule > 0]
    print(f"Running BNSS simulation for {nr} gathers with {len(valid_shooting)} active shot points...")

    pred_blended = simulate_blending(pred_test, valid_shooting, nt, nx)
    pred_test_bnss = pdb_data - (pred_blended - pred_test)

    # Establish data partition indices for training, validation, and testing
    gather_stride_indices = np.arange(2, 256, 3)
    valid_sample_idx = np.round(np.linspace(3, len(gather_stride_indices), 15)).astype(int) - 1
    train_sample_idx = np.setdiff1d(np.arange(len(gather_stride_indices)), valid_sample_idx)

    valid_idx = gather_stride_indices[valid_sample_idx]
    train_idx = np.concatenate(([1], gather_stride_indices[train_sample_idx], [256]))
    test_idx = np.arange(1, nr + 1)

    # Construct feature arrays according to target channel setting
    print(f"Constructing dataset arrays under {args.channels}-CRG formatting...")
    if args.channels == 3:
        hun_train, hun_valid, hun_test = build_3crg_dataset(
            pred_test_bnss, label_data, train_idx, valid_idx, test_idx, nr, nt, nx
        )
    else:
        hun_train, hun_valid, hun_test = build_1crg_dataset(
            pred_test_bnss, train_idx, valid_idx, test_idx
        )

    print(f"Output shapes -> train: {hun_train.shape} | valid: {hun_valid.shape} | test: {hun_test.shape}")

    # Ensure parent folder exists and persist assembled datasets
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sio.savemat(
        output_path,
        {
            "hun_train": hun_train,
            "hun_valid": hun_valid,
            "hun_test": hun_test,
        },
    )
    print(f"Successfully generated and saved BNSS dataset to: {output_path}")

if __name__ == "__main__":
    main()

# ==============================================================================
# Execution Examples
# ==============================================================================
# # Process Step 1 prediction into 3-CRG inputs for Step 2:
# python bnss_dataset_builder.py --step 1 --channels 3
#
# # Process Step 2 prediction into 3-CRG inputs for Step 3:
# python bnss_dataset_builder.py --step 2 --channels 3
#
# # Construct 1-CRG format if needed:
# python bnss_dataset_builder.py --step 1 --channels 1