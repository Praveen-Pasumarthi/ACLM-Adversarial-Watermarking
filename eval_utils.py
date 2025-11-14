import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Statistical Metrics ---

def calculate_statistical_metrics(M_true, M_pred_hard):
    """
    Calculates statistical metrics (BER, TPR, TNR, FNR, FPR) from message arrays.
    M_true and M_pred_hard are hard binary tensors/arrays (0 or 1).
    """
    # Flatten everything to a single 1D array for scikit-learn
    M_true_flat = M_true.cpu().numpy().flatten()
    M_pred_flat = M_pred_hard.cpu().numpy().flatten()
    
    # Calculate confusion matrix: TN, FP, FN, TP
    cm = confusion_matrix(M_true_flat, M_pred_flat, labels=[0, 1])
    
    # Check if the confusion matrix has the expected shape (2x2)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        # Handle cases where only one class (0 or 1) is present or data is empty
        # This will return the simple BER and nan for the rates.
        BER = (M_true_flat != M_pred_flat).mean().item() if len(M_true_flat) > 0 else 0.0
        return {
            'BER': BER,
            'TPR': float('nan'), 'TNR': float('nan'), 'FPR': float('nan'), 'FNR': float('nan'),
            'cm': cm.tolist()
        }

    # Calculate metrics
    P = int(TP + FN) # Actual Positives (1s)
    N = int(TN + FP) # Actual Negatives (0s)

    # Calculate rates with robust zero-division handling
    # We use standard Python float('nan') instead of np.nan for JSON compatibility
    TPR = TP / P if P > 0 else 0.0 # True Positive Rate (Sensitivity/Recall)
    TNR = TN / N if N > 0 else 0.0 # True Negative Rate (Specificity)
    FPR = FP / N if N > 0 else 0.0 # False Positive Rate
    FNR = FN / P if P > 0 else 0.0 # False Negative Rate

    BER = (FP + FN) / (P + N) if (P + N) > 0 else 0.0 # Total bit errors / Total bits (Always reliable since total count is high)

    return {
        'BER': BER,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'cm': cm.tolist()
    }


# --- Attack Simulation (Objective 4 Prep) ---

def simulate_attack_noise(z_prime, strength):
    """Adds small Gaussian noise to the corrupted latent vector for robustness benchmarking."""
    
    # z_prime is the latent vector after the Adversarial Channel (A)
    noise = torch.randn_like(z_prime) * strength
    return z_prime + noise

# --- End of eval_utils.py ---