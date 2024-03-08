from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import argparse
import os


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument("--annealing", type=float, default=0.8)
    parser.add_argument("--unet_path", type=str, default=None)
    parser.add_argument("--text1_path", type=str, default=None)
    parser.add_argument("--text2_path", type=str, default=None)
    parser.add_argument("--target_prompt", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    generator = torch.Generator("cuda").manual_seed(29)
    guidance_scale = 4
    num_inference_steps = 30
    annealing = args.annealing
    unet_weights_all = []
    for i in range(1000):
        unet_weights_all.append((1.0-annealing) / 999 ** 2 * (i - 999) ** 2 + annealing)

    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32)
    unet_path = args.unet_path
    pipe.load_lora_weights(unet_path, adapter_name="lora")
    text1_path = args.text1_path
    pipe.load_lora_weights(text1_path, adapter_name="lora1")
    text2_path = args.text2_path
    pipe.load_lora_weights(text2_path, adapter_name="lora2")
    pipe.to("cuda")

    # Prepare text lora
    pipe.set_adapters_for_text_encoder(['lora1', 'lora2'], pipe.text_encoder, [1.0, -1.0])

    # Encode input prompt
    prompt = args.target_prompt
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    prompt_embeds = pipe.text_encoder(text_input_ids.to(pipe.device), attention_mask=None)
    prompt_embeds = prompt_embeds[0]
    uncond_tokens = ['']
    max_length = prompt_embeds.shape[1]
    uncond_input = pipe.tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    negative_prompt_embeds = pipe.text_encoder(uncond_input.input_ids.to(pipe.device), attention_mask=None)
    negative_prompt_embeds = negative_prompt_embeds[0]
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps

    # Prepare latent variables
    latents = torch.randn([1,4,64,64], generator=generator, device=pipe.device).to(pipe.device)

    # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = {}

    # Add image embeds for IP-Adapter
    added_cond_kwargs = None

    # Optionally get Guidance Scale Embedding
    timestep_cond = None

    # Denoising loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            pipe.unet.set_adapters(['lora'], [unet_weights_all[t.item()]])
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    img = (img * 255).round().squeeze(0)
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save(args.save_path+'test.png')







