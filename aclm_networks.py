import torch
import torch.nn as nn
import torch.nn.functional as F

M_BITS = 256 
Z_LATENT_CHANNELS = 4
Z_LATENT_H = 32 
Z_LATENT_W = 32 

# ----------------------------------------------------------------------
#                           ACLM ENCODER (E)
# ----------------------------------------------------------------------

class ACLMEncoder(nn.Module):
    """
    Encoder (E): Embeds the Codeword (C) into the image latent vector (z).
    Input: Image Latent (z) [B, C, h, w], Codeword (C) [B, M_BITS]
    Output: Watermarked Latent (z_tilde) [B, C, h, w]
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
            nn.Tanh()
        )

    def forward(self, z, C):
        C_map_flat = self.message_map(C)
        C_map = C_map_flat.view(z.size(0), 1, z.size(2), z.size(3))

        z_combined = torch.cat([z, C_map], dim=1)

        watermark_residual = self.conv_layers(z_combined)
        
        z_tilde = z + watermark_residual 
        
        return z_tilde

# ----------------------------------------------------------------------
#                           ACLM DECODER (D)
# ----------------------------------------------------------------------

class ACLMDecoder(nn.Module):
    """
    Decoder (D): Extracts the corrupted Codeword (C_hat) from the attacked latent vector (z').
    Input: Attacked Latent (z_prime) [B, C, h, w]
    Output: Corrupted Codeword (C_hat) [B, M_BITS]
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

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder_head = nn.Sequential(
            nn.Linear(Z_LATENT_CHANNELS * 4, M_BITS * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(M_BITS * 2, M_BITS),
            nn.Sigmoid()
        )

    def forward(self, z_prime):
        features = self.conv_layers(z_prime)

        pooled_features = self.global_pool(features)

        C_hat = self.decoder_head(pooled_features.squeeze())

        return C_hat