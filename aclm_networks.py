import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration Constants (Must be consistent across all files) ---
M_BITS = 448 # Codeword length (Source 256 bits + Parity 192 bits)
Z_LATENT_CHANNELS = 4
Z_LATENT_H = 32 
Z_LATENT_W = 32 

# The size of the flattened message map
FLAT_MAP_SIZE = Z_LATENT_H * Z_LATENT_W # 1024

# The size of the raw flattened latent input (for the simplified decoder)
FLAT_SIZE_INPUT = Z_LATENT_CHANNELS * Z_LATENT_H * Z_LATENT_W # 4096

# ----------------------------------------------------------------------
#                           ACLM ENCODER (E) - (FIXED)
# ----------------------------------------------------------------------

class ACLMEncoder(nn.Module):
    """
    Encoder (E): Embeds the Codeword (C) into the image latent vector (z).
    """
    def __init__(self):
        super(ACLMEncoder, self).__init__()
        
        # 1. Message Embedding: Convert the flat codeword into a spatial map
        self.message_map = nn.Sequential(
            nn.Linear(M_BITS, FLAT_MAP_SIZE),
            nn.LeakyReLU(0.2),
            # CRITICAL FIX: BatchNorm1d forces the signal magnitude to a standard scale
            nn.BatchNorm1d(FLAT_MAP_SIZE), 
            nn.LeakyReLU(0.2),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(Z_LATENT_CHANNELS + 1, Z_LATENT_CHANNELS * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(Z_LATENT_CHANNELS * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(Z_LATENT_CHANNELS * 2, Z_LATENT_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, z, C):
        C_map_flat = self.message_map(C)
        C_map = C_map_flat.view(z.size(0), 1, z.size(2), z.size(3))

        z_combined = torch.cat([z, C_map], dim=1)

        watermark_residual = self.conv_layers(z_combined)
        
        z_tilde = z + watermark_residual 
        
        return z_tilde

# ----------------------------------------------------------------------
#                           ACLM DECODER (D) - (SIMPLIFIED)
# ----------------------------------------------------------------------

class ACLMDecoder(nn.Module):
    """
    Decoder (D): Simplified to linear layers for robust channel establishment.
    """
    def __init__(self):
        super(ACLMDecoder, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.decoder_head = nn.Sequential(
            nn.Linear(FLAT_SIZE_INPUT, M_BITS * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(M_BITS * 2, M_BITS),
            nn.Sigmoid() 
        )

    def forward(self, z_prime):
        flat_features = self.flatten(z_prime)
        C_hat = self.decoder_head(flat_features)
        return C_hat