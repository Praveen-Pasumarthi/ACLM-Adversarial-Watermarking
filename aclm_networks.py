import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration Constants (Must be consistent across all files) ---
M_BITS = 256 
Z_LATENT_CHANNELS = 4
Z_LATENT_H = 32 # Latent height for 256x256 image input (256/8)
Z_LATENT_W = 32 # Latent width for 256x256 image input (256/8)

# ----------------------------------------------------------------------
#                           ACLM ENCODER (E) - (UNCHANGED)
# ----------------------------------------------------------------------

class ACLMEncoder(nn.Module):
    """
    Encoder (E): Embeds the Codeword (C) into the image latent vector (z).
    """
    def __init__(self):
        super(ACLMEncoder, self).__init__()
        
        self.message_map = nn.Sequential(
            nn.Linear(M_BITS, Z_LATENT_H * Z_LATENT_W),
            nn.LeakyReLU(0.2),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(Z_LATENT_CHANNELS + 1, Z_LATENT_CHANNELS * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(Z_LATENT_CHANNELS * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(Z_LATENT_CHANNELS * 2, Z_LATENT_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU() # Kept ReLU to encourage strong initial signal
        )

    def forward(self, z, C):
        C_map_flat = self.message_map(C)
        C_map = C_map_flat.view(z.size(0), 1, z.size(2), z.size(3))

        z_combined = torch.cat([z, C_map], dim=1)

        watermark_residual = self.conv_layers(z_combined)
        
        z_tilde = z + watermark_residual 
        
        return z_tilde

# ----------------------------------------------------------------------
#                         ACLM DECODER (D) - (UPDATED)
# ----------------------------------------------------------------------

# The size after the final convolution layer (Z_LATENT_CHANNELS * 4) * H * W
FLAT_SIZE = Z_LATENT_CHANNELS * 4 * Z_LATENT_H * Z_LATENT_W # 4 * 4 * 32 * 32 = 16384

class ACLMDecoder(nn.Module):
    """
    Decoder (D): Extracts the corrupted Codeword (C_hat) from the attacked latent vector (z').
    """
    def __init__(self):
        super(ACLMDecoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(Z_LATENT_CHANNELS, Z_LATENT_CHANNELS * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(Z_LATENT_CHANNELS * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(Z_LATENT_CHANNELS * 2, Z_LATENT_CHANNELS * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(Z_LATENT_CHANNELS * 4),
            nn.LeakyReLU(0.2),
        )

        # FIX: Replaced Global Pooling with Flatten to avoid signal dilution
        self.flatten = nn.Flatten()

        self.decoder_head = nn.Sequential(
            # Input size is the fully flattened feature map
            nn.Linear(FLAT_SIZE, M_BITS * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(M_BITS * 4, M_BITS),
            nn.Sigmoid()
        )

    def forward(self, z_prime):
        features = self.conv_layers(z_prime)

        # FIX: Use Flatten instead of pooling
        C_hat = self.decoder_head(self.flatten(features))

        return C_hat