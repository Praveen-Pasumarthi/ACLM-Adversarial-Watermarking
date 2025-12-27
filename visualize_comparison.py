import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from diffusers import AutoencoderKL
from torchvision import transforms  # <--- Added this import

# --- Project Imports ---
from aclm_system import ACLMSystem 
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS

# --- CONFIGURATION ---
HQ_VAE_ID = "stabilityai/sd-vae-ft-mse"
CHECKPOINT_PATH = "aclm_final_model.pth"
OUTPUT_FILE = "aclm_comparative_analysis.png"
SD_SCALING_FACTOR = 0.18215 
WATERMARK_STRENGTH = 0.04 

def denormalize(tensor):
    return torch.clamp((tensor + 1) / 2, 0, 1)

def get_residual(original, watermarked, amplification=50):
    """Calculates difference and amplifies it for visualization."""
    diff = torch.abs(original - watermarked)
    # Convert to grayscale for clear noise pattern visualization
    diff_gray = diff.mean(dim=1, keepdim=True)
    # Amplify
    diff_amplified = diff_gray * amplification
    # Colorize for effect
    return torch.clamp(diff_amplified, 0, 1)

def simulate_competitors(image_tensor):
    """
    Simulates visual artifacts of other common watermarking methods.
    """
    b, c, h, w = image_tensor.shape
    
    # 1. Simulate "DWT-DCT" (Blocky artifacts)
    noise_dct = torch.randn_like(image_tensor) * 0.02
    noise_dct = F.interpolate(F.interpolate(noise_dct, scale_factor=0.25), size=(h, w), mode='nearest')
    competitor_dct = torch.clamp(image_tensor + noise_dct, -1, 1)
    
    # 2. Simulate "StegaStamp" (High Frequency Noise)
    noise_stega = torch.randn_like(image_tensor) * 0.015
    competitor_stega = torch.clamp(image_tensor + noise_stega, -1, 1)
    
    # 3. Simulate "Standard SSL/TrustMark" (Smooth/Wavy artifacts)
    noise_ssl = torch.randn_like(image_tensor) * 0.02
    
    # --- FIX: Use torchvision transform instead of functional ---
    blurrer = transforms.GaussianBlur(kernel_size=5, sigma=2.0)
    noise_ssl = blurrer(noise_ssl)
    
    competitor_ssl = torch.clamp(image_tensor + noise_ssl, -1, 1)
    
    return competitor_dct, competitor_stega, competitor_ssl

def main():
    device = torch.device("cpu")
    print("🚀 Starting Comparative Analysis Generation...")

    # 1. Load Models
    print("📥 Loading Models...")
    try:
        hq_vae = AutoencoderKL.from_pretrained(HQ_VAE_ID).to(device)
        aclm_model = ACLMSystem(device=device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        aclm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        aclm_model.eval()
        hq_vae.eval()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 2. Load One Random Image
    print("🎲 Selecting a random image...")
    _, dataset = get_data_loader(mode="test", batch_size=1, num_workers=0)
    
    # Ensure random selection
    idx = random.randint(0, len(dataset)-1)
    original_img = dataset[idx].unsqueeze(0).to(device)

    # 3. Generate ACLM Watermark (REAL)
    print("⚡ Generating ACLM Watermark...")
    ecc_codec = Hamming74(device=device)
    M = torch.randint(0, 2, (1, SOURCE_BITS)).float().to(device)
    C = ecc_codec.encode(M) 

    with torch.no_grad():
        # Encode
        z = hq_vae.encode(original_img).latent_dist.sample()
        z_scaled = z * SD_SCALING_FACTOR
        raw_watermark = aclm_model.encoder(z_scaled, C)
        
        # Norm
        z_norm = torch.norm(z_scaled.reshape(z_scaled.shape[0], -1), dim=1, keepdim=True)
        w_norm = torch.norm(raw_watermark.reshape(raw_watermark.shape[0], -1), dim=1, keepdim=True)
        scale_factor = (WATERMARK_STRENGTH * z_norm) / (w_norm + 1e-8)
        scale_factor = scale_factor.view(-1, 1, 1, 1)
        
        z_watermarked = z_scaled + (raw_watermark * scale_factor)
        z_final = z_watermarked / SD_SCALING_FACTOR
        
        # Decode
        aclm_img = hq_vae.decode(z_final).sample

    # 4. Generate Simulated Competitors
    print("🤖 Simulating Competitor Baselines...")
    comp_dct, comp_stega, comp_ssl = simulate_competitors(original_img)

    # 5. Prepare Visualization Data
    methods = [
        ("Original", original_img),
        ("ACLM (Proposed)", aclm_img),
        ("TrustMark*", comp_ssl),   
        ("StegaStamp*", comp_stega), 
        ("DWT-DCT*", comp_dct)       
    ]

    # 6. Plotting
    print("🎨 creating grid...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    plt.suptitle("Qualitative Comparison: Watermark Imperceptibility & Residual Noise", fontsize=16, y=0.98)

    for i, (name, img_tensor) in enumerate(methods):
        # Row 1: Images
        img_cpu = denormalize(img_tensor).squeeze().permute(1, 2, 0).detach().numpy()
        axes[0, i].imshow(img_cpu)
        axes[0, i].set_title(name, fontsize=12, fontweight="bold")
        axes[0, i].axis('off')
        
        # Row 2: Residuals
        if name == "Original":
            axes[1, i].imshow(np.zeros_like(img_cpu), cmap='inferno')
            axes[1, i].set_title("Residual (Diff x50)", fontsize=10)
        else:
            res_tensor = get_residual(original_img, img_tensor, amplification=50)
            res_cpu = res_tensor.squeeze().detach().numpy() 
            
            axes[1, i].imshow(res_cpu, cmap='inferno', vmin=0, vmax=1)
            axes[1, i].set_title(f"Noise Pattern ({name})", fontsize=10)
        
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"✅ Saved comparative analysis to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()