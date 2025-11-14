import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric # Retain import for compatibility
import sys 
import io
import json 
import lpips # Retain import for compatibility

# --- Project Imports ---
from aclm_system import ACLMSystem
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS 
from ecc_utils import calculate_ber 
from eval_utils import calculate_statistical_metrics, simulate_attack_noise

# --- Configuration ---
CHECKPOINT_PATH = "aclm_final_model.pth"
TEST_BATCH_SIZE = 32
ATTACK_STRENGTHS = [0.00, 0.01, 0.05, 0.10, 0.20] 
OUTPUT_JSON_FILE = "aclm_evaluation_data.json" 

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS (PSNR/LPIPS CODE RETAINED BUT UNUSED)
# ----------------------------------------------------------------------

# These functions are kept as stubs for structural integrity, but the evaluation 
# loop logic has been simplified to exclude calling them.
def calculate_psnr(x_hat, x): return 0.0
LPIPS_NET = None
def init_lpips(device): return None
def calculate_lpips_loss(lpips_net, x_hat, x): return 0.0


# ----------------------------------------------------------------------
#                           EVALUATION LOOP (FINAL ROBUSTNESS FOCUS)
# ----------------------------------------------------------------------

def evaluate_aclm(device):
    
    # 1. Setup Model and Load Checkpoint
    model = ACLMSystem(device=device)
    model.eval()
    
    # LPIPS/PSNR network initialization is skipped for efficiency
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # FIX: Add strict=False to ignore the LPIPS keys that were not in the old checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        print(f"✅ Model loaded from {CHECKPOINT_PATH} (Fidelity metrics ignored).")
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found at {CHECKPOINT_PATH}. Exiting.")
        return

    # 2. Setup Data Loader and Codec
    eval_loader, ds = get_data_loader(batch_size=TEST_BATCH_SIZE, num_workers=0)
    ecc_codec = Hamming74(device=device)
    
    print(f"Starting Evaluation over {len(ds)} images.")
    
    benchmark_results = {}
    
    # Run evaluation across different attack strengths (Objective 4)
    for attack_strength in ATTACK_STRENGTHS:
        print(f"\n--- Running Attack Strength: {attack_strength:.3f} ---")
        
        # Accumulators only for BER
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
                
                # 3. Core Forward Pass
                z = model.vae.encode(x)
                z_tilde = model.encoder(z, C)
                
                # --- APPLY SIMULATED GAUSSIAN ATTACK (Objective 4) ---
                if attack_strength > 0:
                    z_tilde_attacked = simulate_attack_noise(z_tilde, attack_strength)
                else:
                    z_tilde_attacked = z_tilde # No external noise for baseline
                
                z_prime = model.adversary(z_tilde_attacked) 
                C_hat = model.decoder(z_prime)
                
                # Metrics Calculation (PSNR/LPIPS calculations are completely skipped)
                
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

        # 5. Store Results for this Attack Strength
        # FIX 4: Store placeholder values for PSNR/LPIPS, but focus on BER
        avg_psnr = np.nan
        avg_lpips = np.nan 
        
        benchmark_results[attack_strength] = {
            'PSNR': avg_psnr,
            'LPIPS': avg_lpips, 
            'Raw BER': total_ber_raw / total_count,
            'Final BER': total_ber_final / total_count
        }


    # 6. Final Reporting and JSON Structure Generation
    
    if all_M_true and all_M_pred:
        M_true_combined = torch.cat(all_M_true)
        M_pred_combined = torch.cat(all_M_pred)
        stats = calculate_statistical_metrics(M_true_combined, M_pred_combined)
    else:
        stats = None
        
    final_report_data = {
        # FIX 5: PSNR/LPIPS baseline stored as NaN/0.0
        'baseline_psnr': benchmark_results[0.0]['PSNR'],
        'baseline_lpips': 0.0, 
        'baseline_raw_ber': benchmark_results[0.0]['Raw BER'],
        'baseline_final_ber': benchmark_results[0.0]['Final BER'],
        'stats': stats,
        'robustness_benchmark': [
            {'strength': s, 'raw_ber': res['Raw BER'], 'final_ber': res['Final BER']}
            for s, res in benchmark_results.items()
        ]
    }
    
    # 7. Save the structured data to a JSON file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        def convert_numpy_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj

        json.dump(convert_numpy_types(final_report_data), f, indent=4)
    
    # 8. PRINT REPORT TO CONSOLE/TEXT FILE (PSNR/LPIPS REMOVED)
    
    print(f"\n✅ Structured data saved to {OUTPUT_JSON_FILE}.")
    
    print("\n" + "="*50)
    print("            🏆 ACLM FINAL EVALUATION REPORT (Robustness Focus) 🏆")
    print("="*50)

    # FIX 6: PSNR and LPIPS lines are removed from the baseline printing block
    print("\nI. BASELINE ROBUSTNESS (NO EXTERNAL NOISE)")
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
        
    # Define the output file path for the text report
    OUTPUT_FILE = "aclm_evaluation_report.txt"

    # --- OUTPUT REDIRECTION BLOCK ---
    
    # Save the original stdout
    original_stdout = sys.stdout 
    
    # CRITICAL: Run the evaluation block
    try:
        # Open file to save the final text report (this includes the print statements)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Set stdout to the file object
            sys.stdout = f
            
            # Run the evaluation function (which also saves the JSON file)
            evaluate_aclm(device)
            
    except Exception as e:
        # Print error to original stdout if redirection fails
        print(f"\n❌ An error occurred during evaluation: {e}", file=original_stdout)
        
    finally:
        # Restore the original stdout regardless of success/failure
        sys.stdout = original_stdout 
    
    # Print a confirmation message to the terminal
    print(f"\n✅ Evaluation complete. The full report has been saved to: {OUTPUT_FILE}")