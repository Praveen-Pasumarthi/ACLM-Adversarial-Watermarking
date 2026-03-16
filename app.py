import streamlit as st
import subprocess
import glob
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageChops, ImageDraw, ImageEnhance
import aclm_system

@st.cache_resource
def run_thesis_report():
    subprocess.run([sys.executable, "thesis_report_auto.py"])

@st.cache_resource
def load_model():
    model = aclm_system.ACLMSystem()
    model.eval()
    return model

run_thesis_report()
model = load_model()

st.set_page_config(page_title="ACLM Watermarking", layout="wide")
st.title("ACLM Adversarial Watermarking System")

st.subheader("Thesis Report Outputs")

output_images = glob.glob("report_outputs/*.png")

if output_images:
    cols = st.columns(4)
    for idx, img_path in enumerate(output_images[:4]):
        with cols[idx]:
            img = Image.open(img_path)
            st.image(img, width='stretch')

st.divider()

uploaded_file = st.file_uploader("Upload Image to Process", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Input")

    if st.button("Run ACLM Watermarking"):
        w, h = image.size
        
        if w < 256 or h < 256:
            image = image.resize((max(w, 256), max(h, 256)))
            w, h = image.size
            
        left = (w - 256) // 2
        top = (h - 256) // 2
        right = left + 256
        bottom = top + 256
        
        patch = image.crop((left, top, right, bottom))
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(patch).unsqueeze(0)
        
        M = torch.randint(0, 2, (1, 448), dtype=torch.float32)
        
        with torch.no_grad():
            z = model.vae.encode(img_tensor)
            z_tilde = model.encoder(z, M)
            watermarked_tensor = model.vae.decode(z_tilde)
            
            z_extracted = model.vae.encode(watermarked_tensor)
            M_hat = model.decoder(z_extracted)
            
        watermarked_tensor = torch.clamp(watermarked_tensor.squeeze(0), 0, 1)
        watermarked_patch = transforms.ToPILImage()(watermarked_tensor)
        
        output_image = image.copy()
        output_image.paste(watermarked_patch, (left, top))
        
        difference_map = ImageChops.difference(image, output_image)
        
        diff_enhancer = ImageEnhance.Brightness(difference_map)
        diff_bright = diff_enhancer.enhance(25.0)
        
        bg_image = image.convert("L").convert("RGB")
        bg_enhancer = ImageEnhance.Brightness(bg_image)
        bg_dark = bg_enhancer.enhance(0.2)
        
        visual_diff_map = ImageChops.add(bg_dark, diff_bright)
        
        draw = ImageDraw.Draw(output_image)
        draw.rectangle([left, top, right, bottom], outline="red", width=5) 
        
        draw_diff = ImageDraw.Draw(visual_diff_map)
        draw_diff.rectangle([left, top, right, bottom], outline="red", width=5)
        
        st.success("Watermark applied successfully! Location highlighted in red.")
        
        extracted_bits = (M_hat.squeeze() > 0.5).float()
        extracted_grid = extracted_bits.view(16, 28).cpu().numpy() * 255
        extracted_watermark = Image.fromarray(extracted_grid.astype(np.uint8)).resize((256, 256), Image.NEAREST)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(output_image, caption="Watermarked Output Image (Location Highlighted)")
        with col2:
            st.image(visual_diff_map, caption="Watermark Pixel Difference Map (Location Highlighted)")
        with col3:
            st.image(extracted_watermark, caption="Extracted Watermark Proof")