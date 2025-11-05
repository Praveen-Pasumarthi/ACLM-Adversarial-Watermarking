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
    elif len(M_true_flat) == 0:
        return None # No data to report metrics
    else:
        # Handle cases where only one class (0 or 1) is present
        # This is unlikely for random messages, but handles edge cases
        return {
            'BER': (M_true_flat != M_pred_flat).mean(),
            'TPR': np.nan, 'TNR': np.nan, 'FPR': np.nan, 'FNR': np.nan,
            'cm': cm
        }

    # Calculate metrics
    P = TP + FN # Actual Positives (1s)
    N = TN + FP # Actual Negatives (0s)

    TPR = TP / P if P > 0 else np.nan # True Positive Rate (Sensitivity/Recall)
    TNR = TN / N if N > 0 else np.nan # True Negative Rate (Specificity)
    FPR = FP / N if N > 0 else np.nan # False Positive Rate
    FNR = FN / P if P > 0 else np.nan # False Negative Rate

    BER = (FP + FN) / (P + N) # Total bit errors / Total bits

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