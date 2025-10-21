import torch
import torch.optim as optim
import numpy as np
import os
from aclm_system import ACLMSystem, ACLM_Loss, M_BITS
from data_loader import get_data_loader 

# --- Hyperparameters ---
LEARNING_RATE_ED = 1e-4
LEARNING_RATE_A = 1e-5
NUM_EPOCHS = 10
LOG_INTERVAL = 100
IMAGE_HEIGHT = 256

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS
# ----------------------------------------------------------------------

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    print("Starting Minimax ACLM Training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss_ed = 0
        total_loss_a = 0
        total_ber = 0
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            batch_size = x.size(0)
            
            M = generate_random_codeword(batch_size, device)
            
            # ---------------------------------------------------------
            # STEP 1: TRAIN ADVERSARY (A) - MAXIMIZE CORRUPTION
            # ---------------------------------------------------------
            
            with torch.no_grad():
                z = model.vae.encode(x)
                z_tilde = model.encoder(z, M).detach() 

            z_prime = model.adversary(z_tilde)
            M_hat_a = model.decoder(z_prime)
            _, L_A_Total, _, _ = ACLM_Loss(M, M_hat_a, x, x)
            
            optimizer_a.zero_grad()
            L_A_Total.backward()
            optimizer_a.step()
            total_loss_a += L_A_Total.item()
            
            # ---------------------------------------------------------
            # STEP 2: TRAIN ENCODER/DECODER (E/D) - MINIMIZE TOTAL LOSS
            # ---------------------------------------------------------
            
            for param in model.adversary.parameters():
                param.requires_grad = False
            
            M_hat, x_tilde, z_tilde_full = model(x, M)
            
            L_E_D_Total, _, L_Fidelity, L_Recovery = ACLM_Loss(M, M_hat, x, x_tilde)
            
            optimizer_ed.zero_grad()
            L_E_D_Total.backward() 
            optimizer_ed.step()
            
            for param in model.adversary.parameters():
                param.requires_grad = True

            total_loss_ed += L_E_D_Total.item()
            total_ber += calculate_ber(M, M_hat)

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_ber = total_ber / LOG_INTERVAL
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx+1} | "
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