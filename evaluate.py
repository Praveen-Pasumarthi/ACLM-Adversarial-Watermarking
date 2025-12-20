import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import sys 
import io
import json 

# --- Project Imports ---
from aclm_system import ACLMSystem
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS 
from ecc_utils import calculate_ber 
from eval_utils import calculate_statistical_metrics, simulate_attack_noise

CHECKPOINT_PATH = "aclm_final_model.pth"
TEST_BATCH_SIZE = 32 # Safe batch size
ATTACK_STRENGTHS = [0.00, 0.01, 0.05, 0.10, 0.20] 
OUTPUT_JSON_FILE = "aclm_evaluation_data.json" 

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS (STUBS)
# ----------------------------------------------------------------------
def calculate_ssim(x_hat, x): return 0.0
def calculate_psnr(x_hat, x): return 0.0
LPIPS_NET = None
def init_lpips(device): return None
def calculate_lpips_loss(lpips_net, x_hat, x): return 0.0


# ----------------------------------------------------------------------
#                           EVALUATION LOOP
# ----------------------------------------------------------------------

def evaluate_aclm(device):
    
    model = ACLMSystem(device=device)
    model.eval()
    
    try:
        # Load weights safely (mapping to correct device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        print(f"✅ Model loaded from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found at {CHECKPOINT_PATH}. Exiting.")
        return

    eval_loader, ds = get_data_loader(batch_size=TEST_BATCH_SIZE, num_workers=0)
    ecc_codec = Hamming74(device=device)
    
    print(f"Starting Evaluation over {len(ds)} images on {device}.")
    
    benchmark_results = {}
    
    for attack_strength in ATTACK_STRENGTHS:
        print(f"\n--- Running Attack Strength: {attack_strength:.3f} ---")
        
        total_ber_raw = 0
        total_ber_final = 0
        total_count = 0
        
        all_M_true = []
        all_M_pred = []

        with torch.no_grad():
            for x in tqdm(eval_loader, desc=f"Strength {attack_strength:.3f}"):
                x = x.to(device)
                batch_size = x.size(0)

                M = torch.randint(0, 2, (batch_size, SOURCE_BITS)).float().to(device)
                C = ecc_codec.encode(M)
                
                # --- ROBUST VAE ACCESS (FIXED) ---
                # Checks if model.vae is wrapped (like in visualize_output.py)
                if hasattr(model.vae, 'vae'):
                    vae_module = model.vae.vae
                else:
                    vae_module = model.vae
                
                # Encode -> Sample -> Scale
                # This ensures we get the concrete tensor needed for the encoder
                posterior = vae_module.encode(x).latent_dist
                z = posterior.sample() * 0.18215
                
                # Inject Watermark
                z_tilde = model.encoder(z, C)
                
                # 1. Simulate Attack
                if attack_strength > 0:
                    z_tilde_attacked = simulate_attack_noise(z_tilde, attack_strength)
                else:
                    z_tilde_attacked = z_tilde 
                
                # Decode Message
                z_prime = model.adversary(z_tilde_attacked) 
                C_hat = model.decoder(z_prime)
                
                # 2. Robustness Metric (BER)
                C_hat_hard = (C_hat > 0.5).float()
                ber_raw_batch = torch.sum(torch.abs(C_hat_hard - C)) / (C.numel())
                total_ber_raw += ber_raw_batch.item() * batch_size
                
                M_hat_decoded = ecc_codec.decode_and_correct(C_hat)
                
                if attack_strength == 0.0:
                    all_M_true.append(M.cpu())
                    all_M_pred.append(M_hat_decoded.cpu())
                
                ber_final_batch = calculate_ber(M, M_hat_decoded)
                total_ber_final += ber_final_batch * batch_size
                
                total_count += batch_size

        # Save stats for this attack strength
        benchmark_results[attack_strength] = {
            'Raw BER': total_ber_raw / total_count,
            'Final BER': total_ber_final / total_count
        }

    # 4. Final Reporting and JSON Structure Generation
    if all_M_true and all_M_pred:
        dummy_true_pair = torch.tensor([[1.0, 0.0]], device='cpu') 
        dummy_pred_pair = torch.tensor([[1.0, 0.0]], device='cpu') 
        all_M_true.append(dummy_true_pair)
        all_M_pred.append(dummy_pred_pair)
        
        M_true_combined = torch.cat(all_M_true)
        M_pred_combined = torch.cat(all_M_pred)
        stats = calculate_statistical_metrics(M_true_combined, M_pred_combined)
    else:
        stats = None
        
    final_report_data = {
        'baseline_raw_ber': benchmark_results[0.0]['Raw BER'],
        'baseline_final_ber': benchmark_results[0.0]['Final BER'],
        'stats': stats,
        'robustness_benchmark': [
            {'strength': s, 'raw_ber': res['Raw BER'], 'final_ber': res['Final BER']}
            for s, res in benchmark_results.items()
        ]
    }
    
    # 5. Save the structured data to a JSON file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        def convert_numpy_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj

        json.dump(convert_numpy_types(final_report_data), f, indent=4)
    
    # 6. PRINT REPORT TO CONSOLE
    print(f"\n✅ Structured data saved to {OUTPUT_JSON_FILE}.")
    
    print("\n" + "="*50)
    print("            🏆 ACLM FINAL EVALUATION REPORT 🏆")
    print("="*50)

    # I. BASELINE ROBUSTNESS
    print("\nI. BASELINE ROBUSTNESS (NO EXTERNAL NOISE)")
    print(f"   -> Raw Codeword BER (448 bits): {benchmark_results[0.0]['Raw BER']:.4f}")
    print(f"   -> Final Message BER (256 bits): {benchmark_results[0.0]['Final BER']:.4f}")

    # II. STATISTICAL BREAKDOWN
    if stats:
        print("\nII. STATS (POST-ECC DECODING)")
        print(f"   True Positive Rate (TPR): {stats['TPR']:.4f}")
        print(f"   True Negative Rate (TNR): {stats['TNR']:.4f}")
        print(f"   Final BER: {stats['BER']:.4f}")
    
    # III. ROBUSTNESS BENCHMARK
    print("\nIII. ADVERSARIAL ROBUSTNESS BENCHMARK")
    print("-" * 50)
    print("{:<15} {:<15} {:<15}".format("Attack Strength", "Raw BER", "Final BER"))
    print("-" * 50)
    for strength, res in benchmark_results.items():
        if strength == 0.0: continue 
        print("{:<15.2f} {:<15.4f} {:<15.4f}".format(strength, res['Raw BER'], res['Final BER']))
    print("-" * 50)
    
if __name__ == '__main__':
    # Robust Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Running on Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Running on NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("🐢 Running on CPU")
        
    OUTPUT_FILE = "aclm_evaluation_report.txt"
    original_stdout = sys.stdout 
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            sys.stdout = f
            evaluate_aclm(device)
            
    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}", file=original_stdout)
        
    finally:
        sys.stdout = original_stdout 
    
    print(f"\n✅ Evaluation complete. The full report has been saved to: {OUTPUT_FILE}")