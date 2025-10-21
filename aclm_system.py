import torch
import torch.nn as nn
import torch.nn.functional as F
from aclm_networks import ACLMEncoder, ACLMDecoder

# --- Configuration Constants (Check alignment with aclm_networks.py) ---
M_BITS = 256
Z_LATENT_CHANNELS = 4
LAMBDA_FIDELITY = 1.0
LAMBDA_RECOVERY = 1.0
LAMBDA_ADVERSARIAL = 0.5

# ----------------------------------------------------------------------
#                         PLACEHOLDER MODULES
# ----------------------------------------------------------------------

class DummyVAE(nn.Module):
    """
    Placeholder for the Latent Diffusion Model's pre-trained VAE.
    The real VAE (e.g., from Diffusers) will replace this later.
    """
    def __init__(self):
        super().__init__()
        self.scale_factor = 0.18215 

    def encode(self, x):
        """Simulates VAE encoding of an image [B, 3, H, W] to a latent [B, C, h, w]."""
        _, _, H, W = x.shape
        h, w = H // 8, W // 8
        return torch.randn(x.size(0), Z_LATENT_CHANNELS, h, w).to(x.device)

    def decode(self, z):
        """Simulates VAE decoding of a latent [B, C, h, w] back to an image [B, 3, H, W]."""
        H, W = z.shape[2] * 8, z.shape[3] * 8
        return torch.randn(z.size(0), 3, H, W).to(z.device)

class AdversarialChannel(nn.Module):
    """
    The Attacker (A): A small, differentiable network simulating the laundering attack.
    This network is trained to maximize the watermark loss (Minimax objective).
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(Z_LATENT_CHANNELS, Z_LATENT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, z_tilde):
        """Corrupts the watermarked latent vector z_tilde."""
        return self.conv(z_tilde)

# ----------------------------------------------------------------------
#                           ACLM SYSTEM CLASS
# ----------------------------------------------------------------------

class ACLMSystem(nn.Module):
    """
    The unified ACLM framework: combines the VAE, Encoder (E), Decoder (D), and Adversary (A).
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # Core ACLM Networks
        self.encoder = ACLMEncoder().to(device)
        self.decoder = ACLMDecoder().to(device)
        self.adversary = AdversarialChannel().to(device)
        
        # LDM Component 
        self.vae = DummyVAE().to(device) 

    def forward(self, x, M):
        """
        Full forward pass: Embed -> Attack -> Extract.
        :param x: Original Image [B, 3, H, W]
        :param M: Watermark Codeword [B, M_BITS]
        :return: Extracted Codeword (M_hat), Watermarked Image (x_tilde), Watermarked Latent (z_tilde)
        """
        z = self.vae.encode(x)
        
        z_tilde = self.encoder(z, M)
        
        z_prime = self.adversary(z_tilde.detach()) 
        
        M_hat = self.decoder(z_prime)
        
        x_tilde = self.vae.decode(z_tilde)
        
        return M_hat, x_tilde, z_tilde

# ----------------------------------------------------------------------
#                         ACLM MINIMAX LOSS FUNCTION
# ----------------------------------------------------------------------

def ACLM_Loss(M, M_hat, x, x_tilde):
    """
    Computes the three-part Minimax Loss (L_Total) for the ACLM system.
    This function calculates the necessary loss terms, but the final L_Total
    for the Attacker and E/D requires separate optimization steps in the loop.
    """
    
    L_Fidelity = F.mse_loss(x_tilde, x) 

    L_Recovery = F.binary_cross_entropy(M_hat, M)

    L_Adversarial_Term = -L_Recovery 
    
    L_E_D_Total = (LAMBDA_FIDELITY * L_Fidelity) + (LAMBDA_RECOVERY * L_Recovery)
    
    L_A_Total = LAMBDA_ADVERSARIAL * L_Adversarial_Term

    return L_E_D_Total, L_A_Total, L_Fidelity.item(), L_Recovery.item()