import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from ecc_utils import Hamming74, SOURCE_BITS, CODEWORD_BITS
from aclm_system import ACLMSystem, ACLM_Loss
from data_loader import get_data_loader

LEARNING_RATE_ED = 1e-4
LEARNING_RATE_A = 1e-5
NUM_EPOCHS = 10
LOG_INTERVAL = 100
CHECKPOINT_FILENAME = "aclm_final_model.pth"

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS
# ----------------------------------------------------------------------

def generate_random_message(batch_size, device):
    return torch.randint(0, 2, (batch_size, SOURCE_BITS)).float().to(device)

def calculate_ber(M, M_hat):
    M_hat_binary = (M_hat > 0.5).float()
    incorrect_bits = torch.sum(torch.abs(M_hat_binary - M)).item()
    return incorrect_bits / (M.size(0) * M.size(1))

def save_checkpoint(model, optimizer_ed, optimizer_a, epoch, filename):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_ed_state_dict": optimizer_ed.state_dict(),
            "optimizer_a_state_dict": optimizer_a.state_dict(),
        },
        filename,
    )
    print(f"\n✅ Checkpoint saved to {filename} at Epoch {epoch}.")

# ----------------------------------------------------------------------
#                           TRAINING LOOP
# ----------------------------------------------------------------------

def encode_latent(vae, x):

    encoder_out = vae.encode(x)

    if hasattr(encoder_out, "latent_dist"):
        z = encoder_out.latent_dist.sample()
    else:
        z = encoder_out

    return z * 0.18215

def train_aclm():

    device = torch.device("cpu")
    print("Training on CPU")

    # Setup system
    model = ACLMSystem(device=device)
    ecc_codec = Hamming74(device=device)
    train_loader, _ = get_data_loader()

    # Optimizers
    optimizer_ed = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=LEARNING_RATE_ED,
    )

    optimizer_a = optim.Adam(
        model.adversary.parameters(),
        lr=LEARNING_RATE_A,
    )

    model.train()

    print(f"Starting Minimax ACLM Training (Codeword Length: {CODEWORD_BITS} bits)...")

    for epoch in range(1, NUM_EPOCHS + 1):

        total_loss_ed = 0.0
        total_loss_a = 0.0
        total_ber = 0.0

        epoch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            unit="batch",
        )

        for batch_idx, x in enumerate(epoch_iterator):

            x = x.to(device)
            batch_size = x.size(0)

            M = generate_random_message(batch_size, device)
            C = ecc_codec.encode(M)

            # ---------------------------------------------------------
            # STEP 1: TRAIN ADVERSARY
            # ---------------------------------------------------------

            with torch.no_grad():
                z = encode_latent(model.vae, x)
                C_tilde_detached = model.encoder(z, C).detach()

            z_prime = model.adversary(C_tilde_detached)
            C_hat_a = model.decoder(z_prime)
            _, L_A_Total, _, _ = ACLM_Loss(C, C_hat_a)

            optimizer_a.zero_grad()
            L_A_Total.backward()
            optimizer_a.step()

            total_loss_a += L_A_Total.item()

            # ---------------------------------------------------------
            # STEP 2: TRAIN ENCODER / DECODER
            # ---------------------------------------------------------

            for param in model.adversary.parameters():
                param.requires_grad = False

            optimizer_ed.zero_grad()

            with torch.no_grad():
                z = encode_latent(model.vae, x)

            C_tilde = model.encoder(z, C)
            z_prime = model.adversary(C_tilde)
            C_hat = model.decoder(z_prime)

            L_ED_Total, _, _, _ = ACLM_Loss(C, C_hat)

            L_ED_Total.backward()
            optimizer_ed.step()

            for param in model.adversary.parameters():
                param.requires_grad = True

            M_hat_decoded = ecc_codec.decode_and_correct(C_hat.detach())

            total_loss_ed += L_ED_Total.item()
            total_ber += calculate_ber(M, M_hat_decoded)

            # ---------------------------------------------------------
            # LOGGING
            # ---------------------------------------------------------

            if (batch_idx + 1) % LOG_INTERVAL == 0:

                epoch_iterator.write(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx + 1} | "
                    f"E/D Loss: {total_loss_ed / LOG_INTERVAL:.4f} | "
                    f"A Loss: {-total_loss_a / LOG_INTERVAL:.4f} | "
                    f"BER: {total_ber / LOG_INTERVAL:.4f}"
                )

                total_loss_ed = 0.0
                total_loss_a = 0.0
                total_ber = 0.0

    print(" Training complete.")

    save_checkpoint(
        model,
        optimizer_ed,
        optimizer_a,
        NUM_EPOCHS,
        CHECKPOINT_FILENAME,
    )

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    train_aclm()