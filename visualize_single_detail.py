import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

# --- Project Imports ---
from aclm_system import ACLMSystem 
from ecc_utils import Hamming74, SOURCE_BITS

# --- CONFIGURATION ---
HQ_VAE_ID = "stabilityai/sd-vae-ft-mse"
CHECKPOINT_PATH = "aclm_final_model.pth"
OUTPUT_FILE = "aclm_detail_analysis.png"
SD_SCALING_FACTOR = 0.18215 
WATERMARK_STRENGTH = 0.04 
RESIDUAL_AMPLIFICATION = 20 

def denormalize(tensor):
    return torch.clamp((tensor + 1) / 2, 0, 1)

def get_rgb_residual(original, watermarked, amplification=20):
    diff = torch.abs(original - watermarked)
    return torch.clamp(diff * amplification, 0, 1)

def find_high_res_image():
    """
    Scans project folders for a real High-Res image (DIV2K/Flickr2K).
    Bypasses the data_loader's resizing to get raw pixels.
    """
    print("🔍 Hunting for a High-Res source image...")
    
    # Common locations for datasets
    search_paths = ["data", "dataset", "datasets", "DIV2K", "Flickr2K", "test", "."]
    
    candidates = []
    for p in search_paths:
        if os.path.exists(p):
            for root, _, files in os.walk(p):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        try:
                            with Image.open(full_path) as img:
                                w, h = img.size
                                # Must be larger than 512px to allow a good crop
                                if w > 500 and h > 500: 
                                    candidates.append(full_path)
                        except:
                            continue
                if len(candidates) > 50: break 
        if candidates: break
    
    if not candidates:
        print("❌ Could not find any High-Res images (>500px).")
        return None
    
    selected = random.choice(candidates)
    print(f"✅ Found High-Res Source: {selected}")
    return selected

def load_and_crop_image(path):
    """Loads image and takes a sharp 256x256 Center Crop."""
    img = Image.open(path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(512),       # Resize to reasonable scale 
        transforms.CenterCrop(256),   # <--- FIX: Crop ensures high pixel density for faces
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return transform(img).unsqueeze(0)

def main():
    device = torch.device("cpu")
    print("🚀 Starting High-Fidelity Analysis...")

    # 1. Load Models
    hq_vae = AutoencoderKL.from_pretrained(HQ_VAE_ID).to(device)
    aclm_model = ACLMSystem(device=device)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        aclm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except FileNotFoundError:
        print("❌ Checkpoint not found.")
        return
    
    aclm_model.eval(); hq_vae.eval()

    # 2. Get High-Res Data
    img_path = find_high_res_image()
    if img_path:
        original_img = load_and_crop_image(img_path).to(device)
    else:
        return 

    # 3. Watermark
    ecc_codec = Hamming74(device=device)
    M = torch.randint(0, 2, (1, SOURCE_BITS)).float().to(device)
    C = ecc_codec.encode(M) 

    with torch.no_grad():
        z = hq_vae.encode(original_img).latent_dist.sample()
        z_scaled = z * SD_SCALING_FACTOR
        raw_watermark = aclm_model.encoder(z_scaled, C)
        
        z_norm = torch.norm(z_scaled.reshape(1, -1), dim=1, keepdim=True)
        w_norm = torch.norm(raw_watermark.reshape(1, -1), dim=1, keepdim=True)
        scale = (WATERMARK_STRENGTH * z_norm) / (w_norm + 1e-8)
        
        z_final = (z_scaled + raw_watermark * scale.view(-1, 1, 1, 1)) / SD_SCALING_FACTOR
        watermarked_img = hq_vae.decode(z_final).sample

    # 4. Metrics
    mse = torch.mean((denormalize(original_img) - denormalize(watermarked_img)) ** 2)
    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
    print(f"📊 PSNR: {psnr_val:.2f} dB")

    # 5. Visuals
    orig_cpu = denormalize(original_img).squeeze().permute(1, 2, 0).numpy()
    wat_cpu = denormalize(watermarked_img).squeeze().permute(1, 2, 0).numpy()
    
    # Calculate "Dark Mode" Residual
    res_tensor = get_rgb_residual(denormalize(original_img), denormalize(watermarked_img), amplification=RESIDUAL_AMPLIFICATION)
    res_cpu = res_tensor.squeeze().permute(1, 2, 0).numpy()

    # 6. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
    
    axes[0].imshow(orig_cpu)
    axes[0].set_title("Original (Center Crop)", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(wat_cpu)
    axes[1].set_title(f"Watermarked (ACLM)\nPSNR: {psnr_val:.2f} dB", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(res_cpu)
    axes[2].set_title(f"{RESIDUAL_AMPLIFICATION}x Residual (Noise Pattern)", fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Analysis: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()