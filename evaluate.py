import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import sys # <-- ADDED IMPORT
import io  # <-- ADDED IMPORT

# --- Project Imports ---
from aclm_system import ACLMSystem
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS 
# NOTE: calculate_ber must be available. Assuming it's in ecc_utils.
from ecc_utils import calculate_ber 
# Import the new comprehensive metrics and attack utility
from eval_utils import calculate_statistical_metrics, simulate_attack_noise

# --- Configuration ---
CHECKPOINT_PATH = "aclm_final_model.pth"
TEST_BATCH_SIZE = 4
# Attack strengths for Objective 4 benchmarking (Standard Deviations for Gaussian Noise)
ATTACK_STRENGTHS = [0.00, 0.01, 0.05, 0.10, 0.20] 

# ... (calculate_psnr function remains here) ...

def calculate_psnr(x_hat, x):
    """Calculates PSNR metric on a batch of images (tensors)."""
    # Denormalize x_hat and x from [-1, 1] to [0, 1]
    x_hat = (x_hat + 1) / 2
    x = (x + 1) / 2
    
    # Convert to NumPy for skimage calculation
    x_hat = x_hat.permute(0, 2, 3, 1).cpu().numpy()
    x = x.permute(0, 2, 3, 1).cpu().numpy()

    psnr_sum = 0
    for i in range(x.shape[0]):
        # Ensure data types are float for reliable comparison
        psnr_sum += psnr_metric(x[i].astype(np.float32), x_hat[i].astype(np.float32), data_range=1.0)
    return psnr_sum / x.shape[0]


# ----------------------------------------------------------------------
#                           EVALUATION LOOP (UNCHANGED)
# ----------------------------------------------------------------------

def evaluate_aclm(device):
    
    # 1. Setup Model and Load Checkpoint
    model = ACLMSystem(device=device)
    model.eval()
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {CHECKPOINT_PATH}.")
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found at {CHECKPOINT_PATH}. Exiting.")
        return

    # 2. Setup Data Loader and Codec
    eval_loader, ds = get_data_loader(batch_size=TEST_BATCH_SIZE, num_workers=0)
    ecc_codec = Hamming74(device=device)
    
    print(f"Starting Evaluation over {len(ds)} images.")
    
    # Storage for results (Objective 4 Data)
    benchmark_results = {}
    
    # Run evaluation across different attack strengths (Objective 4)
    for attack_strength in ATTACK_STRENGTHS:
        print(f"\n--- Running Attack Strength: {attack_strength:.3f} ---")
        
        total_psnr = 0
        total_ber_raw = 0
        total_ber_final = 0
        total_count = 0
        
        # Lists to gather all bits for a single, overall Confusion Matrix (only needed for 0.0 baseline)
        all_M_true = []
        all_M_pred = []

        with torch.no_grad():
            for x in tqdm(eval_loader, desc=f"Strength {attack_strength:.3f}"):
                x = x.to(device)
                batch_size = x.size(0)

                # Use fixed message M for consistent evaluation
                M = torch.randint(0, 2, (batch_size, SOURCE_BITS)).float().to(device)
                C = ecc_codec.encode(M)
                
                # 3. Core Forward Pass
                z = model.vae.encode(x)
                z_tilde = model.encoder(z, C)
                
                # --- APPLY SIMULATED GAUSSIAN ATTACK (Objective 4) ---
                if attack_strength > 0:
                    z_tilde_attacked = simulate_attack_noise(z_tilde, attack_strength)
                else:
                    z_tilde_attacked = z_tilde # No external noise for baseline
                
                # Forward pass through Adversary (A is always applied) and Decoder (D)
                z_prime = model.adversary(z_tilde_attacked) 
                C_hat = model.decoder(z_prime)
                
                # Metrics Calculation
                
                # 1. Imperceptibility Metric (PSNR only needs to run once, at 0.0 strength)
                if attack_strength == 0.0:
                    x_tilde = model.vae.decode(z_tilde)
                    total_psnr += calculate_psnr(x_tilde, x) * batch_size

                # 2. Robustness Metric (BER)
                
                # Calculate Raw BER (on the 448-bit codeword C)
                C_hat_hard = (C_hat > 0.5).float()
                ber_raw_batch = torch.sum(torch.abs(C_hat_hard - C)) / (C.numel())
                total_ber_raw += ber_raw_batch.item() * batch_size
                
                # Calculate Final BER (on the 256-bit message M after ECC)
                M_hat_decoded = ecc_codec.decode_and_correct(C_hat)
                
                # Gather data for overall confusion matrix (only need one attack level)
                if attack_strength == 0.0:
                    all_M_true.append(M.cpu())
                    # M_hat_decoded is already a hard decision (0 or 1) tensor
                    all_M_pred.append(M_hat_decoded.cpu())
                
                # Calculate final BER using M and M_hat_decoded
                ber_final_batch = calculate_ber(M, M_hat_decoded)
                total_ber_final += ber_final_batch * batch_size
                
                total_count += batch_size

        # 5. Store Results for this Attack Strength
        avg_psnr = total_psnr / total_count if total_psnr > 0 else np.nan
        
        benchmark_results[attack_strength] = {
            'PSNR': avg_psnr,
            'Raw BER': total_ber_raw / total_count,
            'Final BER': total_ber_final / total_count
        }


    # 6. Final Reporting and Confusion Matrix (Objective 1, 2, 4 Fulfillment)
    
    # Calculate Statistical Metrics for the 0.0 baseline (Objective 1)
    if all_M_true and all_M_pred:
        M_true_combined = torch.cat(all_M_true)
        M_pred_combined = torch.cat(all_M_pred)
        stats = calculate_statistical_metrics(M_true_combined, M_pred_combined)
    else:
        stats = None
    
    print("\n" + "="*50)
    print("            🏆 ACLM FINAL EVALUATION REPORT 🏆")
    print("="*50)

    # I. BASELINE METRICS (PSNR, INITIAL BER)
    print("\nI. BASELINE IMPERCEPTIBILITY & UNCORRECTED ROBUSTNESS (NO EXTERNAL NOISE)")
    print(f"   -> Average PSNR (Objective 3): {benchmark_results[0.0]['PSNR']:.2f} dB (Target > 50 dB)")
    print(f"   -> Raw Codeword BER (448 bits): {benchmark_results[0.0]['Raw BER']:.4f}")
    print(f"   -> Final Message BER (256 bits): {benchmark_results[0.0]['Final BER']:.4f}")

    # II. STATISTICAL BREAKDOWN (Objective 1)
    if stats:
        print("\nII. CONFUSION MATRIX & BIT RECOVERY STATS (POST-ECC DECODING)")
        print(f"   True Positive Rate (TPR): {stats['TPR']:.4f}")
        print(f"   True Negative Rate (TNR): {stats['TNR']:.4f}")
        print(f"   Final BER (Target < 0.01): {stats['BER']:.4f}")
        print("\n   Confusion Matrix (True vs Predicted):")
        print(f"   TN | FP\n   FN | TP\n   {np.array(stats['cm'])}")
    
    # III. ROBUSTNESS BENCHMARK (Objective 4)
    print("\nIII. ADVERSARIAL ROBUSTNESS BENCHMARK (BER vs. Simulated Noise)")
    print("   (Data for Robustness Curve Graph)")
    print("-" * 50)
    print("{:<15} {:<15} {:<15}".format("Attack Strength", "Raw BER", "Final BER"))
    print("-" * 50)
    for strength, res in benchmark_results.items():
        if strength == 0.0: continue # Skip baseline since it's above
        print("{:<15.2f} {:<15.4f} {:<15.4f}".format(strength, res['Raw BER'], res['Final BER']))
    print("-" * 50)
    

# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Setup Device Check
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Define the output file path
    OUTPUT_FILE = "aclm_evaluation_report.txt"

    # --- OUTPUT REDIRECTION BLOCK ---
    
    # Save the original stdout
    original_stdout = sys.stdout 
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Set stdout to the file object
            sys.stdout = f
            
            # Run the evaluation function
            evaluate_aclm(device)
            
    except Exception as e:
        # Print error to original stdout if redirection fails
        print(f"\n❌ An error occurred during evaluation: {e}", file=original_stdout)
        
    finally:
        # Restore the original stdout regardless of success/failure
        sys.stdout = original_stdout 
    
    # Print a confirmation message to the terminal
    print(f"\n✅ Evaluation complete. The full report has been saved to: {OUTPUT_FILE}")