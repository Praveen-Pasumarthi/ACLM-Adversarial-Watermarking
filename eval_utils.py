import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Statistical Metrics ---

def calculate_statistical_metrics(M_true, M_pred_hard):
    M_true_flat = M_true.cpu().numpy().flatten()
    M_pred_flat = M_pred_hard.cpu().numpy().flatten()
    
    cm = confusion_matrix(M_true_flat, M_pred_flat, labels=[0, 1])
    
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        BER = (M_true_flat != M_pred_flat).mean().item() if len(M_true_flat) > 0 else 0.0
        return {
            'BER': BER,
            'TPR': float('nan'), 'TNR': float('nan'), 'FPR': float('nan'), 'FNR': float('nan'),
            'cm': cm.tolist()
        }

    # Calculate metrics
    P = int(TP + FN)
    N = int(TN + FP)

    TPR = TP / P if P > 0 else 0.0
    TNR = TN / N if N > 0 else 0.0
    FPR = FP / N if N > 0 else 0.0
    FNR = FN / P if P > 0 else 0.0

    BER = (FP + FN) / (P + N) if (P + N) > 0 else 0.0

    return {
        'BER': BER,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'cm': cm.tolist()
    }

# --- Attack Simulation ---
def simulate_attack_noise(z_prime, strength):
    noise = torch.randn_like(z_prime) * strength
    return z_prime + noise