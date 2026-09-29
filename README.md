<img width="3619" height="1533" alt="figure 3" src="https://github.com/user-attachments/assets/220a5058-c8bb-4167-99d5-49b3d7cc0503" /><h1 align="center">
  Adjacent-CRGs-Multistep-Deblending
</h1>

  **Shicong Lin, Benfeng Wang***. (Tongji University)

<div align="center">
<img width="3619" height="1533" alt="figure 3" src="https://github.com/user-attachments/assets/3a9c378f-be7e-4dd5-9353-9c4eccbe8aaa" />
</div>

# Environment & requirements
Our code is developed and tested on the following hardware:
* CPU: Intel(R) Xeon(R) Gold 5115 @ 2.40GHz
* GPU: NVIDIA GeForce GTX 1080 Ti
  
To ensure exact reproducibility, please use Python 3.8 and install the required dependencies. You can easily set up the environment using the provided `requirements.txt`:
>pip install -r requirements.txt

(Note: Depending on your specific hardware and CUDA version, you may need to install the GPU-compatible version of PyTorch manually via the official PyTorch website.)

# File Description
* :file_folder:**data**: Training, validation, and test datasets of synthetic seismic data.
* :page_facing_up:**train_multistep.py**: Main execution scripts for the multistep deblending process.
* :page_facing_up:**inference_multistep.py**: Standalone inference script for evaluating pre-trained models on designated gathers.
* :page_facing_up:**bnss_dataset_builder.py**: Blending Noise Simulation-Subtraction (BNSS) module for simulating blending noise and assembling adjacent-gather datasets.
* :page_facing_up:**unet_tool.py**: Network architecture of the lightweight U-Net and SNR evaluation metric.
* :page_facing_up:**warmup_tool.py**: Learning rate scheduler featuring warm-up and cosine annealing decay.
* :file_folder:**result**: The optimized model weights from the step1 and step2.

# Workflow
1. **Prepare Dataset**
   Download the required synthetic datasets using the Google Drive link provided in the :file_folder:**data** to obtain the datasets (e.g., dataset_3channel.mat and dataset_1channel.mat).

2. **Multistep Training Pipeline**
**Stage 1 deblending**: 
Train the baseline 3-CRG U-Net model from scratch:
```bash
python train_multistep.py --step 1
``` 
Outputs: Checkpoint and predictions will be saved to `../result/3c1_step1/`.

**Update Dataset (BNSS after step1)**: 
Construct the updated 3-CRG input dataset by subtracting simulated blending noise:
```bash
python bnss_dataset_builder.py --step 1 --channels 3
```
Outputs: Generates `../result/3c1_step1/data_pred_bnss.mat`.

**Stage 2 & 3 deblending with transfer learning**:
Iteratively fine-tune the model using the pre-trained weights and BNSS-updated dataset from the preceding stage:
  
  **Stage 2**:
  ```bash
  python train_multistep.py --step 2
  python bnss_dataset_builder.py --step 2 --channels 3
  ```
  Reads `3c1_step1` outputs and saves Stage 2 results to `../result/3c1_step2/`.

  **Stage 3**:
  ```bash
  python train_multistep.py --step 3
  ```
  Reads `3c1_step2` outputs and saves final results to `../result/3c1_step3/`.

**Quick Run (3 steps)**:
  ```bash
  python train_multistep.py --step 1 && \
  python bnss_dataset_builder.py --step 1 --channels 3 && \
  python train_multistep.py --step 2 && \
  python bnss_dataset_builder.py --step 2 --channels 3 && \
  python train_multistep.py --step 3
  ```
  
**You can also customize settings via command-line arguments**:

**Example: Train Step 2 with custom epochs, learning rate, and batch size**:
  ```bash
  python train_multistep.py --step 2 --epochs 80 --lr 8e-4 --batch_size 4
  ```

**Example: Build a 1-CRG dataset format instead of 3-CRG**:
  ```bash
  python bnss_dataset_builder.py --step 1 --channels 1
  ```

# Standalone Inference
If you already have trained checkpoints and want to run inference separately without re-training:

**Step 1 Model Inference**:
  ```bash
  python inference_multistep.py \
      --step 1 \
      --model ../result/3c1_step1/model.pth \
      --input ../data/dataset_3channel.mat \
      --output ../result/3c1_step1/data_pred.mat
  ```
**Step 2 and Step 3 Model Inference**:
Run inference on the BNSS-updated dataset using the corresponding refined checkpoint (replace `<step>` with `2` or `3`, and `<input>` with `3c1_step1` or `3c1_step2`):

**Example for Step 2**:
  ```bash
  python inference_multistep.py \
      --step 2 \
      --model ../result/3c1_step2/model.pth \
      --input ../result/3c1_step1/data_pred_bnss.mat \
      --output ../result/3c1_step2/data_pred.mat
  ```
