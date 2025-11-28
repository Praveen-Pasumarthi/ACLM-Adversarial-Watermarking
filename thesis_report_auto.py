import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os
import json 
import warnings 

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

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
        
    robustness_df = pd.DataFrame(data['robustness_benchmark'])
    stats = data['stats']
    
    return data, robustness_df, stats

# ----------------------------------------------------------------------
# 2. GENERATE ROBUSTNESS CURVE (BER vs. Attack Strength)
# ----------------------------------------------------------------------
def generate_robustness_curve(df_aclm, df_comp, output_path):
    """Plots ACLM vs. Competitors BER vs. Attack Strength on a single graph."""
    plt.figure(figsize=(10, 6))
    
    if 'strength' not in df_aclm.columns:
         print("Error: ACLM data (strength column) missing from JSON output.")
         return
         
    # --- PLOT 1: Proposed Model (ACLM) Final BER ---
    plt.plot(df_aclm['strength'], df_aclm['final_ber'], 
             marker='o', linestyle='-', linewidth=2, color='darkviolet', 
             label='Proposed Model (ACLM) Final BER')
    
    # --- PLOT 2: Competitors ---
    plt.plot(df_comp['Attack_Strength'], df_comp['InvisMark (Base Paper)'], 
             marker='s', linestyle='-', linewidth=1.5, color='tab:green', 
             label='Existing Model A (Base Paper)')

    plt.plot(df_comp['Attack_Strength'], df_comp['Tree-Ring'], 
             marker='^', linestyle='-', linewidth=1.5, color='sandybrown', 
             label='Existing Model B')
    

    plt.title(r'Adversarial Benchmarking: BER vs. External Attack Strength ($\sigma$)', fontsize=14)
    plt.xlabel(r'Gaussian Attack Strength ($\sigma$)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.axhline(y=0.01, color='r', linestyle='--', linewidth=1, label='Target BER (1%)')
    plt.axhline(y=0.5, color='k', linestyle=':', linewidth=1, label='Random Guessing (0.5)')
    plt.ylim(-0.05, 0.55)
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparative Robustness Curve saved to {output_path}")


# ----------------------------------------------------------------------
# 3. GENERATE CONFUSION MATRIX VISUALIZATION
# ----------------------------------------------------------------------
def generate_confusion_matrix_chart(stats, output_path):
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
    if stats is None:

        stats = { 
            'BER': raw_data['baseline_final_ber']

        } 

    data = {
        'Metric': [
            'Final Message BER (Target < 0.01)', 
            'Raw Codeword BER (Pre-ECC)'
        ],
        'Value': [
            f"{raw_data['baseline_final_ber']:.4f}", 
            
            f"{raw_data['baseline_raw_ber']:.4f}"
        ]
    }
    df_table = pd.DataFrame(data)
    
    with open(output_path, 'w') as f:
        f.write("## Final ACLM Performance Metrics (Robustness Focus)\n")
        f.write(df_table.to_markdown(index=False))
        
    print(f"✅ Final Thesis Table saved to {output_path}")

# ----------------------------------------------------------------------
# 5. GENERATE COMPARISON BAR CHART (Objective 4)
# ----------------------------------------------------------------------
def generate_comparison_charts(competitor_data, output_dir):
    df_comp = pd.DataFrame(competitor_data)
    
    # --- BER Comparison ---
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Model', y='BER', data=df_comp, palette='plasma')
    plt.title('Final Message BER Benchmarking (Robustness)', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.axhline(y=0.01, color='r', linestyle='--', linewidth=1, label='Target BER (1%)')
    plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'comparison_ber.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison BER Chart saved.")


# ----------------------------------------------------------------------
# 6. GENERATE TRAINING HISTORY PLOT (Required Graph)
# ----------------------------------------------------------------------
def generate_training_history_plot(loss_history_data, output_path):
    """
    Plots the E/D Total Loss and BER over the training process.
    """
    df_history = pd.DataFrame(loss_history_data)
    
    x_axis = df_history['Total_Batch']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot 1: Total Loss (Primary Y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Total Training Batches', fontsize=12)
    ax1.set_ylabel('E/D Total Loss', color=color, fontsize=12)
    ax1.plot(x_axis, df_history['E/D_Loss'], color=color, label='E/D Total Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: BER (Secondary Y-axis - Shared X-axis)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('BER (Final)', color=color, fontsize=12)  
    ax2.plot(x_axis, df_history['BER'], color=color, label='Final BER')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.2, 0.55) # Focus on the BER range
    ax2.axhline(y=0.5, color='k', linestyle=':', linewidth=1, label='Random Guessing (0.5)')

    plt.title('ACLM Training History: Loss & BER Convergence', fontsize=14)
    fig.tight_layout() 
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Training History Plot saved to {output_path}")
    
# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
if __name__ == '__main__':
    COMPETITOR_CURVE_DATA = {
        'Attack_Strength': [0.00, 0.01, 0.05, 0.10, 0.20],
        'InvisMark (Base Paper)': [0.2500, 0.2600, 0.2800, 0.3200, 0.3900], 
        'Tree-Ring':        [0.4500, 0.4600, 0.4700, 0.4800, 0.4900]
    }
    COMPETITOR_BAR_BENCHMARKS = [
        {'Model': 'Proposed Model (ACLM)', 'BER': 0.2296},
        {'Model': 'Existing Model A', 'BER': 0.2500},
        {'Model': 'Existing Model B', 'BER': 0.3500},
        {'Model': 'Existing Model C', 'BER': 0.4500}
    ]
    loss_history_data = {
        'Total_Batch': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
                        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        'E/D_Loss': [34.74, 34.70, 34.72, 34.70, 34.70, 34.68, 34.67, 34.62, 34.58, 34.48, 
                     34.19, 33.63, 32.71, 31.86, 31.07, 30.35, 29.65, 29.05, 28.46, 27.92],
        'BER': [0.5005, 0.5026, 0.5003, 0.4984, 0.4954, 0.4937, 0.4900, 0.4830, 0.4776, 0.4639,
                0.4400, 0.4142, 0.3834, 0.3610, 0.3442, 0.3295, 0.3188, 0.3065, 0.2959, 0.2838]
    }

    try:
        raw_data, robustness_df, stats = load_and_prepare_data(DATA_FILE)
        
        print("\n--- Generating Final Project Visualizations and Tables ---")
     
        df_comp_curves = pd.DataFrame(COMPETITOR_CURVE_DATA)

        # 1. Comparative Robustness Curve (Line Graph)
        generate_robustness_curve(robustness_df, df_comp_curves, os.path.join(OUTPUT_DIR, 'robustness_curve_comp.png'))
        
        # 2. Confusion Matrix
        generate_confusion_matrix_chart(stats, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        
        # 3. Final Thesis Table
        generate_thesis_table(raw_data, stats, os.path.join(OUTPUT_DIR, 'thesis_metrics.md'))
        
        # 4. Comparison Bar Chart (Uses the fixed bar chart data)
        generate_comparison_charts(COMPETITOR_BAR_BENCHMARKS, OUTPUT_DIR)
        
        # 5. Training History Plot
        generate_training_history_plot(loss_history_data, os.path.join(OUTPUT_DIR, 'training_history.png'))
        
        print(f"\n--- Output Complete! Check the '{OUTPUT_DIR}/' folder for results. ---")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file '{DATA_FILE}' not found. Please run 'python evaluate.py' first.")
    except Exception as e:
        print(f"\n❌ An error occurred during plotting: {e}")