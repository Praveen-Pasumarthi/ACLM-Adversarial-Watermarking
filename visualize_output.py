import os
# Force CPU for consistency (optional for viz, but safe)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# --- Project Imports ---
from aclm_system import ACLMSystem
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS

# --- CONFIGURATION ---
CHECKPOINT_PATH = "aclm_final_model.pth"
OUTPUT_IMAGE_FILE = "aclm_visual_proof_final.png" 
NUM_IMAGES_TO_SHOW = 4
SD_SCALING_FACTOR = 0.18215 

# TUNING KNOB (Keep low for imperceptibility)
WATERMARK_STRENGTH = 0.04 

def denormalize(tensor):
    return torch.clamp((tensor + 1) / 2, 0, 1)

def visualize_watermarking():
    # Force CPU to match your other scripts
    device = torch.device("cpu")
    print(f"Generating visuals on {device} (using Test Dataset)...")

    # 1. Load Model
    model = ACLMSystem(device=device) 
    model.eval()
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Model loaded from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"❌ Checkpoint {CHECKPOINT_PATH} not found.")
        return

    # 2. Get Data (UPDATED: Uses mode='test')
    # We use mode='test' to prove the model works on unseen images
    loader, _ = get_data_loader(mode="test", batch_size=NUM_IMAGES_TO_SHOW, num_workers=0)
    data_iter = iter(loader)
    original_images = next(data_iter).to(device)

    # 3. Prepare Watermark
    ecc_codec = Hamming74(device=device)
    M = torch.randint(0, 2, (NUM_IMAGES_TO_SHOW, SOURCE_BITS)).float().to(device)
    C = ecc_codec.encode(M) 

    # 4. Process
    with torch.no_grad():
        # Robust VAE Access
        if hasattr(model.vae, 'vae'):
            vae = model.vae.vae
        else:
            vae = model.vae

        # A. Encode to Raw Latents
        encoder_out = vae.encode(original_images)
        if hasattr(encoder_out, "latent_dist"):
             z = encoder_out.latent_dist.sample()
        else:
             z = encoder_out
        
        # B. Baseline Reconstruction
        recon_output = vae.decode(z)
        reconstruction = recon_output.sample if hasattr(recon_output, 'sample') else recon_output

        # C. Watermark Injection (Subtle)
        z_scaled = z * SD_SCALING_FACTOR
        raw_watermark = model.encoder(z_scaled, C)
        
        # Calculate Magnitude for Normalization
        z_norm = torch.norm(z_scaled.reshape(z_scaled.shape[0], -1), dim=1, keepdim=True)
        w_norm = torch.norm(raw_watermark.reshape(raw_watermark.shape[0], -1), dim=1, keepdim=True)
        
        # Apply Stealth Factor
        scale_factor = (WATERMARK_STRENGTH * z_norm) / (w_norm + 1e-8)
        scale_factor = scale_factor.view(-1, 1, 1, 1)
        
        # Inject
        z_watermarked_scaled = z_scaled + (raw_watermark * scale_factor)
        z_final = z_watermarked_scaled / SD_SCALING_FACTOR
        
        # D. Decode
        watermarked_output = vae.decode(z_final)
        watermarked_images = watermarked_output.sample if hasattr(watermarked_output, 'sample') else watermarked_output

    # 5. Visualization
    print("Creating comparison grid...")
    
    orig_cpu = denormalize(original_images).cpu()
    recon_cpu = denormalize(reconstruction).cpu()
    wat_cpu = denormalize(watermarked_images).cpu()
    
    fig, axes = plt.subplots(3, NUM_IMAGES_TO_SHOW, figsize=(15, 9))
    
    for i in range(NUM_IMAGES_TO_SHOW):
        # Original
        axes[0, i].imshow(orig_cpu[i].permute(1, 2, 0).numpy()) 
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original (Test Set)", fontsize=14, loc='left')

        # Reconstruction
        axes[1, i].imshow(recon_cpu[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("VAE Baseline", fontsize=14, loc='left')

        # Watermarked
        axes[2, i].imshow(wat_cpu[i].permute(1, 2, 0).numpy())
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(f"Watermarked (Strength {WATERMARK_STRENGTH})", fontsize=14, loc='left')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"✅ Visual proof saved to {OUTPUT_IMAGE_FILE}")

if __name__ == '__main__':
    visualize_watermarking()