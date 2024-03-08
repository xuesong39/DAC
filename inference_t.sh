CUDA_VISIBLE_DEVICES=0 python inference_t.py \
 --annealing=0.8 \
 --unet_path="U_PATH" \
 --text_path="DELTA_PATH" \
 --target_prompt="xxx" \
 --save_path="./"