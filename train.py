import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm # <--- ADDED TQDM IMPORT
# Ensure ACLM_Loss is imported correctly, assuming it's been updated in aclm_system.py
from aclm_system import ACLMSystem, ACLM_Loss, M_BITS 
from data_loader import get_data_loader 

# --- Hyperparameters ---
LEARNING_RATE_ED = 1e-5
LEARNING_RATE_A = 1e-6 
NUM_EPOCHS = 10
LOG_INTERVAL = 100
IMAGE_HEIGHT = 256

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS (UNCHANGED)
# ----------------------------------------------------------------------
# ... (generate_random_codeword and calculate_ber functions remain here)
def generate_random_codeword(batch_size, device):
    """Generates a batch of random binary codewords (M)."""
    M = torch.randint(0, 2, (batch_size, M_BITS)).float().to(device)
    return M

def calculate_ber(M, M_hat):
    """Calculates the Bit Error Rate (BER) between true and predicted codeword."""
    M_hat_binary = (M_hat > 0.5).float()
    
    incorrect_bits = torch.sum(torch.abs(M_hat_binary - M)).item()
    
    ber = incorrect_bits / (M.size(0) * M.size(1))
    return ber
# ----------------------------------------------------------------------
#                           TRAINING LOOP
# ----------------------------------------------------------------------

def train_aclm():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    model = ACLMSystem(device=device) 
    train_loader, _ = get_data_loader()
    
    optimizer_ed = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=LEARNING_RATE_ED
    )
    optimizer_a = optim.Adam(
        model.adversary.parameters(),
        lr=LEARNING_RATE_A
    )
    
    model.train()

    print("Starting Minimax ACLM Training (Full Adversarial Game)...")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss_ed = 0
        total_loss_a = 0
        total_ber = 0
        
        # --- TQDM PROGRESS BAR ADDED HERE ---
        # Wrap the data loader to display epoch progress visually
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
        
        for batch_idx, x in enumerate(epoch_iterator):
            x = x.to(device)
            batch_size = x.size(0)
            
            M = generate_random_codeword(batch_size, device)
            
            # ---------------------------------------------------------
            # STEP 1: TRAIN ADVERSARY (A) - MAXIMIZE CORRUPTION
            # ---------------------------------------------------------
            # (Logic remains identical to the previous fixed version)
            with torch.no_grad():
                z = model.vae.encode(x)
                z_tilde_detached = model.encoder(z, M).detach()
            
            z_prime = model.adversary(z_tilde_detached)
            M_hat_a = model.decoder(z_prime)
            
            x_dummy = torch.randn_like(x).to(x.device) 
            _, L_A_Total, _, _ = ACLM_Loss(M, M_hat_a, x, x_dummy, z, z_tilde_detached) 
            
            optimizer_a.zero_grad()
            L_A_Total.backward()
            optimizer_a.step()
            total_loss_a += L_A_Total.item()
            
            # ---------------------------------------------------------
            # STEP 2: TRAIN ENCODER/DECODER (E/D) - MINIMIZE TOTAL LOSS
            # ---------------------------------------------------------
            
            for param in model.adversary.parameters():
                param.requires_grad = False
            
            optimizer_ed.zero_grad() 
            
            # 1. Capture original latent z (MUST BE NO_GRAD)
            with torch.no_grad():
                z = model.vae.encode(x)
            
            # 2. Forward Pass for Encoder (GRADIENT ACTIVE!)
            z_tilde = model.encoder(z, M) 
            
            # 3. Apply Adversary (A) - Uses A's fixed weights
            z_prime = model.adversary(z_tilde) 

            # 4. Decode the CORRUPTED latent z_prime
            M_hat = model.decoder(z_prime)
            
            # 5. Decode z_tilde for Fidelity Check (No gradient needed for decoding image)
            with torch.no_grad():
                x_tilde = model.vae.decode(z_tilde)
            
            # Calculate Loss for E/D 
            L_E_D_Total, _, L_Fidelity, L_Recovery = ACLM_Loss(M, M_hat, x, x_tilde, z, z_tilde)
            
            L_E_D_Total.backward() 
            optimizer_ed.step()
            
            for param in model.adversary.parameters():
                param.requires_grad = True

            total_loss_ed += L_E_D_Total.item()
            total_ber += calculate_ber(M, M_hat)

            # ---------------------------------------------------------
            # LOGGING: Using global batch count for consistent logging
            # ---------------------------------------------------------
            current_batch_global = (epoch - 1) * len(train_loader) + batch_idx + 1

            if current_batch_global % LOG_INTERVAL == 0:
                avg_ber = total_ber / LOG_INTERVAL
                
                # Use tqdm's post-logging feature to print the metrics cleanly
                epoch_iterator.write(
                    f"Epoch {epoch}/{NUM_EPOCHS} | Batch {current_batch_global} | "
                    f"E/D Loss: {total_loss_ed/LOG_INTERVAL:.4f} | "
                    f"A Loss: {-total_loss_a/LOG_INTERVAL:.4f} | "
                    f"BER: {avg_ber:.4f}"
                )
                total_loss_ed = 0
                total_loss_a = 0
                total_ber = 0

    print("Training complete.")
    
# --- Execution ---

if __name__ == '__main__':
    train_aclm()