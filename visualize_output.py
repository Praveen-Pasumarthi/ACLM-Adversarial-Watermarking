import os
# Force CPU to avoid VRAM issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from diffusers import AutoencoderKL

# --- Project Imports ---
from aclm_system import ACLMSystem 
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS

# --- CONFIGURATION ---
HQ_VAE_ID = "stabilityai/sd-vae-ft-mse" 
CHECKPOINT_PATH = "aclm_final_model.pth"
OUTPUT_IMAGE_FILE = "aclm_visual_proof_hq.png" 
NUM_IMAGES_TO_SHOW = 4
SD_SCALING_FACTOR = 0.18215 
WATERMARK_STRENGTH = 0.04 

def denormalize(tensor):
    return torch.clamp((tensor + 1) / 2, 0, 1)

def visualize_watermarking():
    device = torch.device("cpu")
    print(f"Generating high-quality visuals on {device}...")
    print("ℹ️ Note: Running at 256x256 (Native Model Resolution)")

    # 1. Load the High-Quality VAE
    print(f"📥 Loading HQ VAE: {HQ_VAE_ID}...")
    try:
        hq_vae = AutoencoderKL.from_pretrained(HQ_VAE_ID).to(device)
        hq_vae.eval()
    except Exception as e:
        print(f"❌ Error loading HQ VAE: {e}")
        return

    # 2. Load YOUR Trained System
    print("📥 Loading your trained ACLM System...")
    model = ACLMSystem(device=device)
    model.eval()
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Trained model loaded from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"❌ Checkpoint {CHECKPOINT_PATH} not found.")
        return

    # 3. Get Test Data (WITH RANDOM SAMPLING)
    print("🎲 Selecting random images from Test Set...")
    _, test_dataset = get_data_loader(mode="test", batch_size=NUM_IMAGES_TO_SHOW, num_workers=0)
    
    total_images = len(test_dataset)
    # Randomly pick 4 images
    random_indices = random.sample(range(total_images), NUM_IMAGES_TO_SHOW)
    print(f"   Indices selected: {random_indices}")
    
    batch_images = []
    for idx in random_indices:
        batch_images.append(test_dataset[idx])
    
    original_images = torch.stack(batch_images).to(device)

    # 4. Prepare Watermark Message
    ecc_codec = Hamming74(device=device)
    M = torch.randint(0, 2, (NUM_IMAGES_TO_SHOW, SOURCE_BITS)).float().to(device)
    C = ecc_codec.encode(M) 

    # 5. Process
    with torch.no_grad():
        # A. Encode to Latents using HQ VAE
        z = hq_vae.encode(original_images).latent_dist.sample()
        
        # B. Baseline Reconstruction (HQ VAE)
        recon_output = hq_vae.decode(z)
        reconstruction = recon_output.sample

        # C. Watermark Injection
        z_scaled = z * SD_SCALING_FACTOR
        raw_watermark = model.encoder(z_scaled, C)
        
        # --- Stealth Magnitude Normalization ---
        z_norm = torch.norm(z_scaled.reshape(z_scaled.shape[0], -1), dim=1, keepdim=True)
        w_norm = torch.norm(raw_watermark.reshape(raw_watermark.shape[0], -1), dim=1, keepdim=True)
        scale_factor = (WATERMARK_STRENGTH * z_norm) / (w_norm + 1e-8)
        scale_factor = scale_factor.view(-1, 1, 1, 1)
        
        # Apply scaled watermark
        z_watermarked_scaled = z_scaled + (raw_watermark * scale_factor)
        z_final = z_watermarked_scaled / SD_SCALING_FACTOR
        
        # D. Decode to Image (HQ VAE)
        watermarked_output = hq_vae.decode(z_final)
        watermarked_images = watermarked_output.sample

    # 6. Visualization
    print("Creating comparison grid...")
    
    orig_cpu = denormalize(original_images).cpu()
    recon_cpu = denormalize(reconstruction).cpu()
    wat_cpu = denormalize(watermarked_images).cpu()
    
    fig, axes = plt.subplots(3, NUM_IMAGES_TO_SHOW, figsize=(15, 9))
    
    for i in range(NUM_IMAGES_TO_SHOW):
        # Row 1: Original
        axes[0, i].imshow(orig_cpu[i].permute(1, 2, 0).numpy()) 
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original", fontsize=14, loc='left')

        # Row 2: VAE Baseline (HQ)
        axes[1, i].imshow(recon_cpu[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"Reconstruction (HQ VAE)", fontsize=14, loc='left')

        # Row 3: Watermarked (Your Model + HQ VAE)
        axes[2, i].imshow(wat_cpu[i].permute(1, 2, 0).numpy())
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(f"Watermarked (ACLM)", fontsize=14, loc='left')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"✅ High-quality visual proof saved to {OUTPUT_IMAGE_FILE}")

if __name__ == '__main__':
    visualize_watermarking()