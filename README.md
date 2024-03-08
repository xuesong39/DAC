# Doubly Abductive Counterfactual Inference for Text-based Image Editing

This respository contains the code for the CVPR 2024 paper [Doubly Abductive Counterfactual Inference for Text-based Image Editing](https://arxiv.org/abs/2403.02981).

## Setup

### Dependency Installation

First, clone the repository:

```bash
git clone https://github.com/xuesong39/DAC
```

Then, install the dependencies in a new virtual environment:

```bash
cd DAC
git clone https://github.com/huggingface/diffusers -b v0.24.0
cd diffusers
pip install -e .
```

Finally, cd in the main folder `DAC` and run:
```bash
pip install -r requirements.txt
```

### Data Preparation

The images and annotations we use in the paper can be found [here](https://drive.google.com/drive/folders/12ueFwPhrJ7ncssJLeYP6n304Z7v5492j?usp=drive_link).
For the format of data used in the experiments, we provide some examples in the folder `DAC/data`. For example, for the image `DAC/data/cat/train/cat.jpeg`, the folder containing source prompt is `DAC/data/cat/` while that containing target prompt is `DAC/data/cat-cap/`.


## Usage

### Abduction-1
The fine-tuning script for abduction on _U_ is `train_text_to_image_lora.sh` as follows:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="ORIGIN_DATA_PATH"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=1000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=512 \
  --output_dir="U_PATH" \
  --validation_prompt="xxx" \
  --report_to="wandb" \
  --validation_epochs=500
```
Please specify `TRAIN_DIR` (e.g., "./data/cat/"), `--output_dir` (e.g., "ckpt/cat"), and `--validation_prompt` (e.g., "A cat.").

### Abduction-2
The fine-tuning script for abduction on Δ is `train_text_to_image_lora_t.sh` as follows:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="TARGET_DATA_PATH"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_text_to_image_lora_t.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --unet_lora_path="U_PATH" \
  --train_data_dir=$TRAIN_DIR --caption_column="text" \
  --resolution=512 --train_text_encoder \
  --train_batch_size=1 \
  --num_train_epochs=1000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --annealing=0.8 \
  --output_dir="DELTA_PATH" \
  --report_to="wandb" \
  --validation_epochs=500
```
Please specify `TRAIN_DIR` (e.g., "./data/cat-cap/"), `--unet_lora_path` (e.g., "ckpt/cat"), and `--output_dir` (e.g., "ckpt/cat-cap-annealing0.8"). You can also change `--annealing` to achieve control on hyperparameter $\eta$.

### Action & Prediction
The inference script is `inference_t.sh` as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_t.py \
 --annealing=0.8 \
 --unet_path="U_PATH" \
 --text_path="DELTA_PATH" \
 --target_prompt="xxx" \
 --save_path="./"
```
Please specify `--unet_path` (e.g., "ckpt/cat"), `--text_path` (e.g., "ckpt/cat-cap-annealing0.8"), and `--target_prompt` (e.g., "A cat wearing a wool cap.").

## Optional Usage
This part contains the implementation mentioned in the ablation analysis section in the paper, i.e., ablation on Abduction-1. We could incorporate another exogenous variable _T_ in the Abduction-1 to further improve fidelity.

### Abduction-1
The fine-tuning script for abduction on _U_ is the same as the above.

The fine-tuning script for abduction on _T_ is `train_text_to_image_lora_t.sh` as follows:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="ORIGIN_DATA_PATH"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_text_to_image_lora_t.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --unet_lora_path="U_PATH" \
  --train_data_dir=$TRAIN_DIR --caption_column="text" \
  --resolution=512 --train_text_encoder \
  --train_batch_size=1 \
  --num_train_epochs=1000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --annealing=0.8 \
  --output_dir="T_PATH" \
  --report_to="wandb" \
  --validation_epochs=500
```
Please specify `TRAIN_DIR` (e.g., "./data/cat/"), `--unet_lora_path` (e.g., "ckpt/cat"), and `--output_dir` (e.g., "ckpt/cat-annealing0.8")

### Abduction-2
The fine-tuning script for abduction on Δ is `train_text_to_image_lora_t2.sh` as follows:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="TARGTE_DATA_PATH"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_text_to_image_lora_t2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --unet_lora_path="U_PATH" \
  --text_lora1_path="T_PATH" \
  --train_data_dir=$TRAIN_DIR --caption_column="text" \
  --resolution=512 --train_text_encoder \
  --train_batch_size=1 \
  --num_train_epochs=1000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --annealing=0.8 \
  --output_dir="DELTA_PATH" \
  --report_to="wandb" \
  --validation_epochs=500
```
Please specify `TRAIN_DIR` (e.g., "./data/cat-cap/"), `--unet_lora_path` (e.g., "ckpt/cat"), `--text_lora1_path` (e.g., "ckpt/cat-annealing0.8"), and `--output_dir` (e.g., "ckpt/cat-cap-annealing0.8-t2").

### Action & Prediction
The inference script is `inference_t2.sh` as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_t2.py \
 --annealing=0.8 \
 --unet_path="U_PATH" \
 --text1_path="T_PATH" \
 --text2_path="DELTA_PATH" \
 --target_prompt="xxx" \
 --save_path="./"
```
Please specify `--unet_path` (e.g., "ckpt/cat"), `--text1_path` (e.g., "ckpt/cat-annealing0.8"), `--text2_path` (e.g., "ckpt/cat-cap-annealing0.8-t2"), and `--target_prompt` (e.g., "A cat wearing a wool cap.").

## Checkpoints

We provide some checkpoints in the following:

| Image | Abduction-1 | Abduction-2 |
|:---:|:---:|:---:|
|`DAC/data/cat`| [_U_](https://drive.google.com/drive/folders/12UFolxqxQDlHDOZZiQZoK5cEwxxyWjZS?usp=drive_link) |[Δ](https://drive.google.com/drive/folders/1Sbc3dSqY2KyhQ7KRxwwoT1KQcrf7Kz5q?usp=drive_link)|
|`DAC/data/glass`| [_U_](https://drive.google.com/drive/folders/1wf1XbC3Gi5Nv5ig-zNlY1xAVuWsV-5Rz?usp=drive_link) |[Δ](https://drive.google.com/drive/folders/1TZ9KAvbB5_XDHN7VcH-0gtVW5sa-gF0h?usp=drive_link)|
|`DAC/data/black`| [_U_](https://drive.google.com/drive/folders/1G2pIboOJMr2iirmsqATyMQcHpCYeXNGy?usp=drive_link) |[Δ](https://drive.google.com/drive/folders/17CiCbOtCJR8jgvQ8Ut-UDA1v-XuTxaVv?usp=drive_link)|
|`DAC/data/cat`| [_U_](https://drive.google.com/drive/folders/12UFolxqxQDlHDOZZiQZoK5cEwxxyWjZS?usp=drive_link), [_T_](https://drive.google.com/drive/folders/1XD23tfc2nANpoPjMW4t8u09mdjg_zBCR?usp=drive_link) |[Δ](https://drive.google.com/drive/folders/1AuQuUNE5Q9GZXRXggdk7psccTv10P5ZX?usp=drive_link)|
|`DAC/data/glass`|[_U_](https://drive.google.com/drive/folders/1wf1XbC3Gi5Nv5ig-zNlY1xAVuWsV-5Rz?usp=drive_link), [_T_](https://drive.google.com/drive/folders/1lRwgxacYpmenI-SjrEZTbiApXm3Vhukz?usp=drive_link) |[Δ](https://drive.google.com/drive/folders/1PiWoaCpnuf2xbhGTvx6gatINtbcSgbF6?usp=drive_link)|
|`DAC/data/black`|[_U_](https://drive.google.com/drive/folders/1G2pIboOJMr2iirmsqATyMQcHpCYeXNGy?usp=drive_link), [_T_](https://drive.google.com/drive/folders/17sxSfJNfutDN9luIh6OoU2BnxHMUO4Jd?usp=drive_link) |[Δ](https://drive.google.com/drive/folders/1dml7yAuNa84pSNT6hp3ErvyP5RqNtlxt?usp=drive_link)|

## Acknowledgments

In this code we refer to the following codebase: [Diffusers](https://github.com/huggingface/diffusers) and [PEFT](https://github.com/huggingface/peft). Great thanks to them!

