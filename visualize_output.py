import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid

# --- Project Imports ---
from aclm_system import ACLMSystem
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS

# --- CONFIGURATION ---
CHECKPOINT_PATH = "aclm_final_model.pth"
OUTPUT_IMAGE_FILE = "aclm_visual_proof.png"
NUM_IMAGES_TO_SHOW = 4
SD_SCALING_FACTOR = 0.18215 

def denormalize(tensor):
    """Converts a [-1, 1] tensor back to [0, 1] for plotting, with robust clamping."""
    return torch.clamp((tensor + 1) / 2, 0, 1)

def visualize_watermarking():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating visual samples on {device}...")

    # 1. Load Model
    model = ACLMSystem(device=device)
    model.eval()
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Checkpoint {CHECKPOINT_PATH} not found.")
        return

    # 2. Get Data
    loader, _ = get_data_loader(batch_size=NUM_IMAGES_TO_SHOW, num_workers=0)
    data_iter = iter(loader)
    original_images = next(data_iter).to(device)

    # 3. Prepare Watermark
    ecc_codec = Hamming74(device=device)
    M = torch.randint(0, 2, (NUM_IMAGES_TO_SHOW, SOURCE_BITS)).float().to(device)
    C = ecc_codec.encode(M) 

    # 4. Process
    with torch.no_grad():
        if hasattr(model.vae, 'vae'):
            vae = model.vae.vae
        else:
            vae = model.vae

        # A. Encode to Raw Latents
        posterior = vae.encode(original_images).latent_dist
        z = posterior.sample()

        # B. VAE Reconstruction (Baseline)
        recon_output = vae.decode(z)
        reconstruction = recon_output.sample if hasattr(recon_output, 'sample') else recon_output

        # C. Watermark Injection (NORMALIZED RESIDUAL)
        # 1. Scale input for Encoder
        z_scaled = z * SD_SCALING_FACTOR
        
        # 2. Get Raw Watermark Signal
        # This signal is currently too "loud" (high magnitude)
        raw_watermark = model.encoder(z_scaled, C)
        
        # 3. Calculate Magnitudes
        # We calculate the average L2 norm (strength) of the latent vs the watermark
        z_norm = torch.norm(z_scaled.reshape(z_scaled.shape[0], -1), dim=1, keepdim=True)
        w_norm = torch.norm(raw_watermark.reshape(raw_watermark.shape[0], -1), dim=1, keepdim=True)
        
        # 4. Normalize Watermark to be relative to the Image Content
        # We want the watermark to be epsilon * ImageStrength
        epsilon = 0.08  # The "Stealth Factor". 0.08 = 8% perturbation (Visible but subtle). 
                        # Lower this to 0.05 or 0.03 for "Invisible".
        
        # Avoid division by zero
        scale_factor = (epsilon * z_norm) / (w_norm + 1e-8)
        
        # Reshape scale factor to broadcast: [B, 1, 1, 1]
        scale_factor = scale_factor.view(-1, 1, 1, 1)
        
        # 5. Inject Scaled Watermark
        z_watermarked_scaled = z_scaled + (raw_watermark * scale_factor)
        
        # 6. Unscale for Decoder
        z_final = z_watermarked_scaled / SD_SCALING_FACTOR
        
        # D. Decode Watermarked Latents
        watermarked_output = vae.decode(z_final)
        watermarked_images = watermarked_output.sample if hasattr(watermarked_output, 'sample') else watermarked_output

    # 5. Visualization
    print("Creating comparison grid...")
    
    orig_cpu = denormalize(original_images).cpu()
    recon_cpu = denormalize(reconstruction).cpu()
    wat_cpu = denormalize(watermarked_images).cpu()
    
    fig, axes = plt.subplots(3, NUM_IMAGES_TO_SHOW, figsize=(15, 9))
    
    for i in range(NUM_IMAGES_TO_SHOW):
        # Row 1: Original
        ax1 = axes[0, i]
        ax1.imshow(orig_cpu[i].permute(1, 2, 0).numpy()) 
        ax1.axis('off')
        if i == 0: ax1.set_title("Original Input", fontsize=14, loc='left')

        # Row 2: VAE Reconstruction
        ax2 = axes[1, i]
        ax2.imshow(recon_cpu[i].permute(1, 2, 0).numpy())
        ax2.axis('off')
        if i == 0: ax2.set_title("VAE Reconstruction (Baseline)", fontsize=14, loc='left')

        # Row 3: Watermarked Output
        ax3 = axes[2, i]
        ax3.imshow(wat_cpu[i].permute(1, 2, 0).numpy())
        ax3.axis('off')
        if i == 0: ax3.set_title("Watermarked (Normalized Injection)", fontsize=14, loc='left')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"✅ Visual proof saved to {OUTPUT_IMAGE_FILE}")
    print("   (The bottom row should now look almost identical to the middle row.)")

if __name__ == '__main__':
    visualize_watermarking()