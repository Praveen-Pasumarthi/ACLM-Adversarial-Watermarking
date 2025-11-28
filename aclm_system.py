import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL 
from aclm_networks import ACLMEncoder, ACLMDecoder 

# --- Configuration Constants ---
M_BITS = 448
Z_LATENT_CHANNELS = 4
LAMBDA_RECOVERY = 10.0
LAMBDA_ADVERSARIAL = 0.5

class AdversarialChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(Z_LATENT_CHANNELS, Z_LATENT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, z_tilde):
        return self.conv(z_tilde)
    
# ----------------------------------------------------------------------
#                         REAL VAE INTEGRATION
# ----------------------------------------------------------------------

class RealVAE(nn.Module):
    def __init__(self):
        super().__init__()
        print("📥 Loading pre-trained VAE from Stable Diffusion v1-5...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.scale_factor = 0.18215

    def encode(self, x):
        latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.scale_factor

    def decode(self, z):
        z = z / self.scale_factor
        image = self.vae.decode(z).sample
        return image

# ----------------------------------------------------------------------
#                           ACLM SYSTEM CLASS
# ----------------------------------------------------------------------

class ACLMSystem(nn.Module):
  
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.encoder = ACLMEncoder().to(device)
        self.decoder = ACLMDecoder().to(device)
        self.adversary = AdversarialChannel().to(device)
        
        self.vae = RealVAE().to(device) 
        self.vae.eval() 
        
        for param in self.vae.parameters():
            if param.requires_grad:
                raise Exception("VAE parameters are NOT frozen!")

    def forward(self, x, M):
       
        z = self.vae.encode(x)
        z_tilde = self.encoder(z, M)
        z_prime = self.adversary(z_tilde.detach())
        M_hat = self.decoder(z_prime)
        
        return M_hat, z_tilde, z # M_hat (for loss), z_tilde/z (for analysis/input to E/A)

# ----------------------------------------------------------------------
#                 ACLM MINIMAX LOSS FUNCTION
# ----------------------------------------------------------------------

def ACLM_Loss(M, M_hat): 
  
    L_Recovery = F.binary_cross_entropy(M_hat, M)
    L_Adversarial_Term = -L_Recovery 
    L_E_D_Total = (LAMBDA_RECOVERY * L_Recovery)
    L_A_Total = LAMBDA_ADVERSARIAL * L_Adversarial_Term
    
    return L_E_D_Total, L_A_Total, 0.0, L_Recovery.item()