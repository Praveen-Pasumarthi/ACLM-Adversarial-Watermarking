import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
# Import ECC components
from ecc_utils import Hamming74, SOURCE_BITS, CODEWORD_BITS 
# Ensure ACLM_Loss is imported correctly, assuming it's been updated in aclm_system.py
from aclm_system import ACLMSystem, ACLM_Loss, M_BITS 
from data_loader import get_data_loader 

# --- Hyperparameters ---
# Final stable configuration after tuning
LEARNING_RATE_ED = 1e-4 
LEARNING_RATE_A = 1e-5 
NUM_EPOCHS = 10
LOG_INTERVAL = 100
IMAGE_HEIGHT = 256
CHECKPOINT_FILENAME = "aclm_final_model.pth" 

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS (UNCHANGED)
# ----------------------------------------------------------------------

def generate_random_message(batch_size, device):
    """Generates a batch of random SOURCE_BITS (256) messages M."""
    M = torch.randint(0, 2, (batch_size, SOURCE_BITS)).float().to(device)
    return M

def calculate_ber(M, M_hat):
    """Calculates the Final Bit Error Rate (BER) between original message (M) and decoded message (M_hat)."""
    M_hat_binary = (M_hat > 0.5).float()
    
    incorrect_bits = torch.sum(torch.abs(M_hat - M)).item()
    
    ber = incorrect_bits / (M.size(0) * M.size(1))
    return ber

def save_checkpoint(model, optimizer_ed, optimizer_a, epoch, filename):
    """Saves model and optimizer states."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_ed_state_dict': optimizer_ed.state_dict(),
        'optimizer_a_state_dict': optimizer_a.state_dict(),
    }, filename)
    print(f"\n✅ Checkpoint saved to {filename} at Epoch {epoch}.")

# ----------------------------------------------------------------------
#                           TRAINING LOOP (FINAL)
# ----------------------------------------------------------------------

def train_aclm():
    # Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # Initialize Model and ECC Codec
    model = ACLMSystem(device=device) 
    ecc_codec = Hamming74(device=device) 
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

    print(f"Starting Minimax ACLM Training (Codeword Length: {CODEWORD_BITS} bits)...")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss_ed = 0
        total_loss_a = 0
        total_ber = 0
        
        # Wrap the data loader to display epoch progress visually
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
        
        for batch_idx, x in enumerate(epoch_iterator):
            x = x.to(device)
            batch_size = x.size(0)
            
            # --- ECC INTEGRATION: Encode M into the longer codeword C ---
            M = generate_random_message(batch_size, device) 
            C = ecc_codec.encode(M) 
            
            # ---------------------------------------------------------
            # STEP 1: TRAIN ADVERSARY (A) - MAXIMIZE CORRUPTION
            # ---------------------------------------------------------
            
            with torch.no_grad():
                z = model.vae.encode(x)
                C_tilde_detached = model.encoder(z, C).detach() 
            
            z_prime = model.adversary(C_tilde_detached)
            C_hat_a = model.decoder(z_prime) 
            
            # FIX: ACLM_Loss simplified (only C and C_hat_a passed)
            _, L_A_Total, _, _ = ACLM_Loss(C, C_hat_a) 
            
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
            
            with torch.no_grad():
                z = model.vae.encode(x) # Z is input for Encoder
            
            C_tilde = model.encoder(z, C) 
            z_prime = model.adversary(C_tilde) 
            C_hat = model.decoder(z_prime) 
            
            # FIX: ACLM_Loss simplified (only C and C_hat passed)
            L_E_D_Total, _, L_Fidelity, L_Recovery = ACLM_Loss(C, C_hat) 
            
            L_E_D_Total.backward() 
            optimizer_ed.step()
            
            for param in model.adversary.parameters():
                param.requires_grad = True
            
            # --- BER CALCULATION: Decode Noisy Codeword (C_hat) back to Message (M) ---
            M_hat_decoded = ecc_codec.decode_and_correct(C_hat.detach())
            
            total_loss_ed += L_E_D_Total.item()
            total_ber += calculate_ber(M, M_hat_decoded) 

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_ber = total_ber / LOG_INTERVAL
                
                # Use tqdm's post-logging feature to print the metrics cleanly
                epoch_iterator.write(
                    f"Epoch {epoch}/{NUM_EPOCHS} | Batch {(epoch - 1) * len(train_loader) + batch_idx + 1} | "
                    f"E/D Loss: {total_loss_ed/LOG_INTERVAL:.4f} | "
                    f"A Loss: {-total_loss_a/LOG_INTERVAL:.4f} | "
                    f"BER: {avg_ber:.4f} (Final)"
                )
                total_loss_ed = 0
                total_loss_a = 0
                total_ber = 0

    print("Training complete.")
    
    # ---------------------------------------------------------
    # FINAL STEP: SAVE CHECKPOINT
    # ---------------------------------------------------------
    save_checkpoint(model, optimizer_ed, optimizer_a, NUM_EPOCHS, CHECKPOINT_FILENAME)
    
# --- Execution ---

if __name__ == '__main__':
    train_aclm()