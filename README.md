# AgriMamba-Guided Multimodal Framework with Pathology-aware Alignment for Plant Disease Severity Grading



## Project Overview

This repository contains the implementation of a two-stage training framework for plant disease analysis. The model utilizes the VMamba architecture to achieve high performance in segmentation and grading tasks.

---

## 1. Dependencies

The project runs on a Linux environment with the following specifications. You must match these versions to ensure compatibility with the custom CUDA kernels.


  ```bash
  conda create -n your_env_name 
  conda activate your_env_name
  pip install -r reqirements.txt

  ```

---

## 2. Environment Setup

The VMamba model requires a specific kernel for the selective scan operation. You need to compile this component manually on your local machine.

```bash
# build kernel for VMamba dependencies：
cd selective_scan && pip install .

```


The framework requires a pretrained CLIP backbone for feature extraction. You must download the weight files from [https://huggingface.co/openai/clip-vit-large-patch14/tree/main]. Place the downloaded files into the following directory:
pretrain/clip/clip-vit-large-patch14





pretrain checkpoint

- download from [link](https://drive.google.com/file/d/1O9P6XLuWtUxFa70vwrYCRVedRAFutczV/view?usp=sharing) (This is the VMamba checkpoint pretrained on ImageNet. We re-upload it since the original VMamba link has broken.)
- Create a ``pretrain/`` folder and place the checkpoint in it



## 3. Dataset
Regarding the dataset acquisition, please contact us at [http://agrimamba.samlab.cn/] to obtain the necessary files.


---

## 4. Training Pipeline

The training process follows a sequential two-stage strategy. You must save the outputs of the first stage to initialize the second stage.

### Step 1: LMLS

This initial stage focuses on the primary features of the plant leaves. The model trains for 50 epochs with a small learning rate.

```bash
python train_stage1.py \
--output-dir ./output/stage1 \
--epochs 50 \
--batch-size 8 \
--data-set dataset_10350 \
--lr 3e-5

```

### Step 2: Generate RGB image of leaves

This step filters out the complex background, leaving only the target leaves


```bash
python segmentation_model/generate_stage1_results.py \
    --checkpoint ./output/stage1/best_stage1_checkpoint.pth \
    --data-set dataset_10350 \
    --output-dir ./output/stage1/stage1_out_img \
    --batch-size 16

```

### Step 3: TMLS

This stage refines the results by focusing on specific lesion areas. It uses the visual outputs from Stage I as a reference for the new training cycle.

```bash
python train_stage2.py \
--stage1-results ./output/stage1/stage1_out_img \
--data-set dataset_10350 \
--output-dir ./output/stage2 \
--epochs 100 \
--batch-size 8 \
--lr 3e-5

```

---

## 5. Evaluation

The evaluation script combines the best checkpoints from both stages. It processes the test set and saves intermediate visualization results for analysis.

```bash

python evaluate.py \
    --leaf-checkpoint output/stage1/best_stage1_checkpoint.pth \
    --lesion-checkpoint output/stage2/best_stage2_checkpoint.pth \
    --data-set dataset_10350 \
    --output-dir ./output/evaluation \
    --save-intermediate


```



---


