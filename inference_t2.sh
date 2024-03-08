CUDA_VISIBLE_DEVICES=0 python inference_t2.py \
 --annealing=0.8 \
 --unet_path="U_PATH" \
 --text1_path="T_PATH" \
 --text2_path="DELTA_PATH" \
 --target_prompt="xxx" \
 --save_path="./"