import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Ensure this is installed: pip install seaborn
import numpy as np
import os
import json 

# --- CONFIGURATION ---
DATA_FILE = "aclm_evaluation_data.json"
OUTPUT_DIR = "report_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. LOAD DATA FROM JSON
# ----------------------------------------------------------------------
def load_and_prepare_data(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
        
    # Prepare Robustness DataFrame
    robustness_df = pd.DataFrame(data['robustness_benchmark'])
    
    # Prepare Statistical Data
    stats = data['stats']
    
    return data, robustness_df, stats


# ----------------------------------------------------------------------
# 2. GENERATE ROBUSTNESS CURVE (Required Graph)
# ----------------------------------------------------------------------
def generate_robustness_curve(df, output_path):
    """Plots BER vs. Attack Strength."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['strength'], df['final_ber'], 
             marker='o', linestyle='-', color='tab:red', 
             label='Final Message BER (Post-ECC)')
    
    plt.plot(df['strength'], df['raw_ber'], 
             marker='x', linestyle='--', color='tab:blue', alpha=0.6,
             label='Raw Codeword BER (Pre-ECC)')

    # FIX 1: Use raw strings for titles to prevent SyntaxWarning
    plt.title(r'ACLM Adversarial Robustness: BER vs. Simulated Gaussian Attack ($\sigma$)', fontsize=14)
    plt.xlabel(r'Gaussian Attack Strength ($\sigma$)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.axhline(y=0.5, color='k', linestyle=':', linewidth=1, label='Random Guessing (0.5)')
    plt.ylim(-0.05, 0.55)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Robustness Curve saved to {output_path}")


# ----------------------------------------------------------------------
# 3. GENERATE CONFUSION MATRIX VISUALIZATION
# ----------------------------------------------------------------------
def generate_confusion_matrix_chart(stats, output_path):
    """Plots the confusion matrix."""
    # FIX 2: Check if statistical data (stats) exists AND has the 'cm' key
    if not stats or 'cm' not in stats:
        print("Warning: Cannot generate Confusion Matrix, statistical data is missing or corrupted.")
        return

    cm_data = np.array(stats['cm'])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_data, annot=True, fmt=".4f", cmap="Blues", 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'], 
                cbar=False, linewidths=.5)
    plt.title('Normalized Confusion Matrix (Post-ECC Decoded Message)', fontsize=12)
    plt.ylabel('Actual Bits (M)', fontsize=10)
    plt.xlabel('Predicted Bits (M\' - decoded)', fontsize=10)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion Matrix Chart saved to {output_path}")


# ----------------------------------------------------------------------
# 4. GENERATE FINAL THESIS TABLE
# ----------------------------------------------------------------------
def generate_thesis_table(raw_data, stats, output_path):
    """Generates the final statistical metrics table."""
    
    # FIX 3: Provide dummy values if statistical data (stats) is None to prevent 'NoneType' crash
    if stats is None:
        stats = { 'BER': np.nan, 'TPR': np.nan, 'TNR': np.nan } 

    data = {
        'Metric': [
            'Avg. PSNR (Imperceptibility)', 
            'Final Message BER (Target < 0.01)', 
            'Raw Codeword BER (Pre-ECC)', 
            'True Positive Rate (TPR)', 
            'True Negative Rate (TNR)'
        ],
        'Value': [
            f"{raw_data['baseline_psnr']:.2f} dB (FAILURE)",
            f"{stats['BER']:.4f}",
            f"{raw_data['baseline_raw_ber']:.4f}",
            f"{stats['TPR']:.4f}",
            f"{stats['TNR']:.4f}"
        ]
    }
    df_table = pd.DataFrame(data)
    
    with open(output_path, 'w') as f:
        f.write("## Final ACLM Performance Metrics\n")
        f.write(df_table.to_markdown(index=False))
        
    print(f"✅ Final Thesis Table saved to {output_path}")


# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    try:
        raw_data, robustness_df, stats = load_and_prepare_data(DATA_FILE)
        
        print("\n--- Generating Final Project Visualizations and Tables ---")
        
        # Run all generation functions
        generate_robustness_curve(robustness_df, os.path.join(OUTPUT_DIR, 'robustness_curve.png'))
        generate_confusion_matrix_chart(stats, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        generate_thesis_table(raw_data, stats, os.path.join(OUTPUT_DIR, 'thesis_metrics.md'))
        
        print(f"\n--- Output Complete! Check the '{OUTPUT_DIR}/' folder for results. ---")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file '{DATA_FILE}' not found. Please run 'python evaluate.py' first.")
    except Exception as e:
        print(f"\n❌ An error occurred during plotting: {e}")