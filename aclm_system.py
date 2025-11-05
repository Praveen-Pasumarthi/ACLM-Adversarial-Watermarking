import torch
import torch.nn as nn
import torch.nn.functional as F
# Import VAE component from diffusers
from diffusers import AutoencoderKL 
from aclm_networks import ACLMEncoder, ACLMDecoder 

# --- Configuration Constants (Check alignment with aclm_networks.py) ---
M_BITS = 448
Z_LATENT_CHANNELS = 4
# We keep the aggressive lambda values here to ensure recovery priority continues
LAMBDA_FIDELITY = 0.01    # Fidelity on Latent Space (MSE)
LAMBDA_RECOVERY = 50.0     # Recovery on Codeword (BCE)
LAMBDA_ADVERSARIAL = 0.5  # Adversarial Attack Weight

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
#                         REAL VAE INTEGRATION
# ----------------------------------------------------------------------

class RealVAE(nn.Module):
    """
    Real VAE component loaded from Stable Diffusion v1-5.
    """
    def __init__(self):
        super().__init__()
        print("📥 Loading pre-trained VAE from Stable Diffusion v1-5...")
        # Load the pre-trained VAE weights
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )
        # CRITICAL: Freeze the VAE parameters entirely
        self.vae.requires_grad_(False)
        self.scale_factor = 0.18215 # Standard SD VAE scale factor

    def encode(self, x):
        """Encodes image [B, 3, H, W] to latent [B, 4, h, w]."""
        # x is assumed to be in the [-1, 1] range (from data_loader)
        latent = self.vae.encode(x).latent_dist.sample()
        # Scale the latent output
        return latent * self.scale_factor

    def decode(self, z):
        """Decodes latent to image space."""
        # Un-scale the latent input
        z = z / self.scale_factor
        image = self.vae.decode(z).sample
        # The output image is in [-1, 1] range
        return image

# ----------------------------------------------------------------------
#                           ACLM SYSTEM CLASS (UPDATED VAE)
# ----------------------------------------------------------------------

class ACLMSystem(nn.Module):
    """
    The unified ACLM framework: combines the Real VAE, Encoder (E), Decoder (D), and Adversary (A).
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # Core ACLM Networks
        self.encoder = ACLMEncoder().to(device)
        self.decoder = ACLMDecoder().to(device)
        self.adversary = AdversarialChannel().to(device)
        
        # LDM Component (Now Real)
        self.vae = RealVAE().to(device) # Replaced DummyVAE with RealVAE
        self.vae.eval() # Set VAE to evaluation mode
        
        # Sanity check: ensure VAE parameters are truly frozen
        for param in self.vae.parameters():
            if param.requires_grad:
                raise Exception("VAE parameters are NOT frozen!")

    def forward(self, x, M):
        """
        Full forward pass: Embed -> Attack -> Extract.
        """
        # Encode image to get original latent vector z
        z = self.vae.encode(x)
        
        # Embed Watermark (E)
        z_tilde = self.encoder(z, M)
        
        # Apply Adversarial Channel (A)
        z_prime = self.adversary(z_tilde.detach()) 
        
        # Extract Watermark (D)
        M_hat = self.decoder(z_prime)
        
        # Decode Watermarked Latent (z_tilde) for pixel fidelity (optional in this fix)
        x_tilde = self.vae.decode(z_tilde)
        
        # Returns the necessary components for loss calculation
        return M_hat, x_tilde, z_tilde, z 

# ----------------------------------------------------------------------
#                 UPDATED ACLM MINIMAX LOSS FUNCTION (UNCHANGED)
# ----------------------------------------------------------------------

def ACLM_Loss(M, M_hat, x, x_tilde, z, z_tilde):
    """
    Computes the three-part Minimax Loss (L_Total) using Latent Space Fidelity.
    """
    
    # 1. Fidelity Loss (L_Fidelity): Latent MSE
    L_Fidelity = F.mse_loss(z_tilde, z) 

    # 2. Recovery Loss (L_Recovery): Codeword BCE
    L_Recovery = F.binary_cross_entropy(M_hat, M)

    # 3. Adversarial Loss (L_Adversarial): Robustness
    L_Adversarial_Term = -L_Recovery 
    
    # Total Loss for Encoder/Decoder (minimized)
    L_E_D_Total = (LAMBDA_FIDELITY * L_Fidelity) + (LAMBDA_RECOVERY * L_Recovery)
    
    # Total Loss for Attacker (maximized)
    L_A_Total = LAMBDA_ADVERSARIAL * L_Adversarial_Term

    return L_E_D_Total, L_A_Total, L_Fidelity.item(), L_Recovery.item()