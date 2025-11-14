import torch
import torch.nn as nn
import torch.nn.functional as F
# Import VAE component from diffusers
from diffusers import AutoencoderKL 
from aclm_networks import ACLMEncoder, ACLMDecoder 

# --- Configuration Constants (Check alignment with aclm_networks.py) ---
M_BITS = 448
Z_LATENT_CHANNELS = 4
# LAMBDA_FIDELITY is removed as it is no longer used.
LAMBDA_RECOVERY = 10.0     # Keeping high priority for recovery
LAMBDA_ADVERSARIAL = 0.5   # Adversarial Attack Weight

class AdversarialChannel(nn.Module):
    """
    The Attacker (A): A small, differentiable network simulating the laundering attack.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(Z_LATENT_CHANNELS, Z_LATENT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, z_tilde):
        """Corrupts the watermarked latent vector z_tilde."""
        return self.conv(z_tilde)
    
# ----------------------------------------------------------------------
#                         REAL VAE INTEGRATION (UNCHANGED)
# ----------------------------------------------------------------------

class RealVAE(nn.Module):
    """
    Real VAE component loaded from Stable Diffusion v1-5.
    """
    def __init__(self):
        super().__init__()
        print("📥 Loading pre-trained VAE from Stable Diffusion v1-5...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.scale_factor = 0.18215 # Standard SD VAE scale factor

    def encode(self, x):
        """Encodes image [B, 3, H, W] to latent [B, 4, h, w]."""
        latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.scale_factor

    def decode(self, z):
        """Decodes latent to image space."""
        z = z / self.scale_factor
        image = self.vae.decode(z).sample
        return image

# ----------------------------------------------------------------------
#                           ACLM SYSTEM CLASS (SIMPLIFIED FOR ROBUSTNESS)
# ----------------------------------------------------------------------

class ACLMSystem(nn.Module):
    """
    The unified ACLM framework, focused purely on robustness.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.encoder = ACLMEncoder().to(device)
        self.decoder = ACLMDecoder().to(device)
        self.adversary = AdversarialChannel().to(device)
        
        self.vae = RealVAE().to(device) 
        self.vae.eval() 
        
        # Sanity check: ensure VAE parameters are truly frozen
        for param in self.vae.parameters():
            if param.requires_grad:
                raise Exception("VAE parameters are NOT frozen!")

    def forward(self, x, M):
        """
        Full forward pass: Embed -> Attack -> Extract.
        FIX 1: Removed x_tilde from return, as it is only for Fidelity/PSNR.
        """
        # Encode image to get original latent vector z
        z = self.vae.encode(x)
        
        # Embed Codeword (E)
        z_tilde = self.encoder(z, M)
        
        # Apply Adversarial Channel (A)
        z_prime = self.adversary(z_tilde.detach()) 
        
        # Extract Watermark (D)
        M_hat = self.decoder(z_prime)
        
        # Returns only the components strictly required for the robustness loss and further steps
        return M_hat, z_tilde, z # M_hat (for loss), z_tilde/z (for analysis/input to E/A)

# ----------------------------------------------------------------------
#                 ACLM MINIMAX LOSS FUNCTION (SIMPLIFIED FOR ROBUSTNESS)
# ----------------------------------------------------------------------

# FIX 2: Loss function simplified to accept only required arguments and focus on L_Recovery.
def ACLM_Loss(M, M_hat): 
    """
    Computes the Minimax Loss (L_Total) focused solely on L_Recovery.
    """
    
    # 1. Fidelity Loss (L_Fidelity): REMOVED ENTIRELY

    # 2. Recovery Loss (L_Recovery): Codeword BCE
    L_Recovery = F.binary_cross_entropy(M_hat, M)

    # 3. Adversarial Loss (L_Adversarial): Robustness
    L_Adversarial_Term = -L_Recovery 
    
    # Total Loss for Encoder/Decoder (minimized)
    # FIX 3: L_E_D_Total relies ONLY on L_Recovery.
    L_E_D_Total = (LAMBDA_RECOVERY * L_Recovery)
    
    # Total Loss for Attacker (maximized)
    L_A_Total = LAMBDA_ADVERSARIAL * L_Adversarial_Term

    # FIX 4: Return tuple simplified. L_Fidelity is now 0.0.
    return L_E_D_Total, L_A_Total, 0.0, L_Recovery.item()