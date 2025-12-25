import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import numpy as np
from tqdm import tqdm
import sys
import json

# --- Project Imports ---
from aclm_system import ACLMSystem
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS
from ecc_utils import calculate_ber
from eval_utils import calculate_statistical_metrics, simulate_attack_noise

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

CHECKPOINT_PATH = "aclm_final_model.pth"
TEST_BATCH_SIZE = 32
ATTACK_STRENGTHS = [0.00, 0.01, 0.05, 0.10, 0.20]
OUTPUT_JSON_FILE = "aclm_evaluation_data.json"
OUTPUT_FILE = "aclm_evaluation_report.txt"

# ----------------------------------------------------------------------
#                       ROBUST LATENT ENCODING
# ----------------------------------------------------------------------

def encode_latent(vae, x):
    encoder_out = vae.encode(x)
    if hasattr(encoder_out, "latent_dist"):
        z = encoder_out.latent_dist.sample()
    else:
        z = encoder_out
    return z * 0.18215

# ----------------------------------------------------------------------
#                           EVALUATION LOOP
# ----------------------------------------------------------------------

def evaluate_aclm(device):

    model = ACLMSystem(device=device)
    model.eval()

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"✅ Model loaded from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # --------------------------------------------------------------
    # LOAD TEST DATA ONLY
    # --------------------------------------------------------------
    eval_loader, test_dataset = get_data_loader(
        mode="test",
        batch_size=TEST_BATCH_SIZE,
        num_workers=0
    )

    print(f"📊 Testing images loaded: {len(test_dataset)}")

    ecc_codec = Hamming74(device=device)
    benchmark_results = {}

    for attack_strength in ATTACK_STRENGTHS:

        print(f"\n--- Running Attack Strength: {attack_strength:.3f} ---")

        total_ber_raw = 0.0
        total_ber_final = 0.0
        total_count = 0

        all_M_true = []
        all_M_pred = []

        with torch.no_grad():
            for x in tqdm(eval_loader, desc=f"Strength {attack_strength:.3f}"):

                x = x.to(device)
                batch_size = x.size(0)

                M = torch.randint(0, 2, (batch_size, SOURCE_BITS)).float().to(device)
                C = ecc_codec.encode(M)

                z = encode_latent(model.vae, x)
                z_tilde = model.encoder(z, C)

                if attack_strength > 0:
                    z_tilde = simulate_attack_noise(z_tilde, attack_strength)

                z_prime = model.adversary(z_tilde)
                C_hat = model.decoder(z_prime)

                C_hat_hard = (C_hat > 0.5).float()
                ber_raw = torch.sum(torch.abs(C_hat_hard - C)) / C.numel()
                total_ber_raw += ber_raw.item() * batch_size

                M_hat = ecc_codec.decode_and_correct(C_hat)
                ber_final = calculate_ber(M, M_hat)
                total_ber_final += ber_final * batch_size

                if attack_strength == 0.0:
                    all_M_true.append(M.cpu())
                    all_M_pred.append(M_hat.cpu())

                total_count += batch_size

        benchmark_results[attack_strength] = {
            "Raw BER": total_ber_raw / total_count,
            "Final BER": total_ber_final / total_count,
        }

    # --------------------------------------------------------------
    # STATISTICS (NO-ATTACK CASE)
    # --------------------------------------------------------------

    if all_M_true and all_M_pred:
        M_true_all = torch.cat(all_M_true)
        M_pred_all = torch.cat(all_M_pred)
        stats = calculate_statistical_metrics(M_true_all, M_pred_all)
    else:
        stats = None

    final_report_data = {
        "baseline_raw_ber": benchmark_results[0.0]["Raw BER"],
        "baseline_final_ber": benchmark_results[0.0]["Final BER"],
        "stats": stats,
        "robustness_benchmark": [
            {
                "strength": s,
                "raw_ber": res["Raw BER"],
                "final_ber": res["Final BER"],
            }
            for s, res in benchmark_results.items()
        ],
    }

    # --------------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------------

    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(convert_numpy(final_report_data), f, indent=4)

    # --------------------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------------------

    print("\n" + "=" * 50)
    print("        🏆 ACLM FINAL EVALUATION REPORT 🏆")
    print("=" * 50)

    print("\nBASELINE (NO ATTACK)")
    print(f"Raw BER   : {benchmark_results[0.0]['Raw BER']:.6f}")
    print(f"Final BER : {benchmark_results[0.0]['Final BER']:.6f}")

    print("\nROBUSTNESS BENCHMARK")
    print("-" * 50)
    print("{:<15} {:<15} {:<15}".format("Attack", "Raw BER", "Final BER"))
    print("-" * 50)

    for strength, res in benchmark_results.items():
        if strength == 0.0:
            continue
        print(
            "{:<15.2f} {:<15.6f} {:<15.6f}".format(
                strength, res["Raw BER"], res["Final BER"]
            )
        )

    print("-" * 50)
    print(f"\n✅ JSON saved to {OUTPUT_JSON_FILE}")

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cpu")
    print("🐢 Running evaluation on CPU")

    original_stdout = sys.stdout

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            sys.stdout = f
            evaluate_aclm(device)
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}", file=original_stdout)
    finally:
        sys.stdout = original_stdout

    print(f"\n✅ Evaluation complete. Report saved to {OUTPUT_FILE}")