import torch
import torch.nn.functional as F
from aclm_system import ACLMSystem, ACLM_Loss, M_BITS 
from data_loader import get_data_loader
from ecc_utils import Hamming74, SOURCE_BITS
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# --- Configuration ---
CHECKPOINT_PATH = "aclm_checkpoint.pth"
TEST_BATCH_SIZE = 4

def calculate_psnr(x_hat, x):
    """Calculates PSNR metric on a batch of images (tensors)."""
    # Denormalize x_hat and x from [-1, 1] to [0, 1]
    x_hat = (x_hat + 1) / 2
    x = (x + 1) / 2
    
    # Convert to NumPy for skimage calculation
    x_hat = x_hat.permute(0, 2, 3, 1).cpu().numpy()
    x = x.permute(0, 2, 3, 1).cpu().numpy()

    psnr_sum = 0
    for i in range(x.shape[0]):
        psnr_sum += psnr_metric(x[i], x_hat[i], data_range=1.0)
    return psnr_sum / x.shape[0]

def evaluate_aclm(device):
    
    # 1. Setup Model and Load Checkpoint
    model = ACLMSystem(device=device)
    model.eval()
    
    # Load model weights from a checkpoint (assuming one was saved by train.py)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {CHECKPOINT_PATH}.")
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found at {CHECKPOINT_PATH}. Exiting.")
        return

    # 2. Setup Data Loader for Evaluation (we can use the training set for now)
    eval_loader, ds = get_data_loader(batch_size=TEST_BATCH_SIZE, num_workers=0)
    ecc_codec = Hamming74(device=device)

    total_psnr = 0
    total_ber = 0
    total_count = 0
    
    print(f"Starting Evaluation over {len(ds)} images...")

    with torch.no_grad():
        for x in tqdm(eval_loader, desc="Evaluating Model"):
            x = x.to(device)
            batch_size = x.size(0)
            
            # Use fixed message M for consistent evaluation
            M = torch.randint(0, 2, (batch_size, SOURCE_BITS)).float().to(device)
            C = ecc_codec.encode(M)
            
            # Forward pass through E, A, D
            z = model.vae.encode(x)
            z_tilde = model.encoder(z, C)
            z_prime = model.adversary(z_tilde) 
            C_hat = model.decoder(z_prime)

            # Metrics Calculation
            
            # 1. Imperceptibility Metric (PSNR)
            x_tilde = model.vae.decode(z_tilde)
            total_psnr += calculate_psnr(x_tilde, x) * batch_size

            # 2. Robustness Metric (BER)
            M_hat_decoded = ecc_codec.decode_and_correct(C_hat)
            total_ber += calculate_ber(M, M_hat_decoded) * batch_size
            
            total_count += batch_size

    # 3. Final Results
    avg_psnr = total_psnr / total_count
    avg_ber = total_ber / total_count

    print("\n--- Evaluation Results (ACLM) ---")
    print(f"Average PSNR (Imperceptibility): {avg_psnr:.2f} dB")
    print(f"Average BER (Robustness): {avg_ber:.4f}")
    print("---------------------------------")


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    evaluate_aclm(device)