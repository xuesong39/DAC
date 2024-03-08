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