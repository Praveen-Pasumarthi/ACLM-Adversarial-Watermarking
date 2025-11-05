import torch
import torch.nn.functional as F
import numpy as np

# --- Configuration Constants ---
# R = Redundancy bits (3)
# K = Data bits (4)
# N = Codeword length (7)
HAMMING_N = 7
HAMMING_K = 4
HAMMING_R = 3
BLOCK_SIZE = HAMMING_K

# Your total desired message length is 256 bits.
# We will use 256 bits as the source message M, and the codeword C will be longer.
SOURCE_BITS = 256
BLOCKS = SOURCE_BITS // HAMMING_K # Number of 4-bit blocks (256 / 4 = 64 blocks)
CODEWORD_BITS = BLOCKS * HAMMING_N # Total Codeword length (64 * 7 = 448 bits)

# Update M_BITS in other files (aclm_networks.py, aclm_system.py) to 448
# This is the length of the codeword (C) that the Encoder receives.
# ACLM_M_BITS should be 448.

class Hamming74:
    """
    Implements simple batch-wise Hamming (7,4) encoding and decoding.
    
    NOTE: This implementation is for demonstration. Real-world ECC uses 
    optimized matrix multiplication (generator matrix G, parity-check matrix H) 
    which is faster than bitwise ops, but this is clearer for initial testing.
    """

    def __init__(self, device='cpu'):
        self.device = device
        
        # Generator matrix G (4x7) for encoding 4 data bits into a 7-bit codeword [d1 d2 d3 d4 p1 p2 p3]
        # p1=d1+d2+d4, p2=d1+d3+d4, p3=d2+d3+d4 (mod 2)
        # Note: We use a simplified parity structure here.
        
        # Generator Matrix (Not explicitly used below, but defined for understanding)
        # H matrix defines parity checks.
        
    def _encode_single_block(self, d):
        """Encodes one 4-bit data block d into one 7-bit codeword c (mod 2)."""
        d = d.long().tolist() # Convert to list of 0s and 1s
        d1, d2, d3, d4 = d[0], d[1], d[2], d[3]
        
        # Calculate parity bits (Even Parity)
        p1 = d1 ^ d2 ^ d4 
        p2 = d1 ^ d3 ^ d4 
        p3 = d2 ^ d3 ^ d4 
        
        # Codeword c = [d1, d2, d3, d4, p1, p2, p3]
        return torch.tensor([d1, d2, d3, d4, p1, p2, p3], dtype=torch.float32, device=self.device)

    def encode(self, M):
        """
        Encodes a batch of SOURCE_BITS (256) messages M into CODEWORD_BITS (448) codewords C.
        M is [B, 256]. C is [B, 448].
        """
        batch_size = M.size(0)
        M_reshaped = M.view(batch_size, BLOCKS, BLOCK_SIZE) # [B, 64, 4]
        
        C = []
        for i in range(batch_size):
            codeword_batch = []
            for j in range(BLOCKS):
                codeword_batch.append(self._encode_single_block(M_reshaped[i, j, :]))
            C.append(torch.cat(codeword_batch))
            
        return torch.stack(C) # [B, 448]

    def decode_and_correct(self, C_hat):
        """
        Decodes a batch of noisy codewords C_hat (448) and returns corrected M_hat (256).
        C_hat is [B, 448]. M_hat is [B, 256].
        """
        batch_size = C_hat.size(0)
        C_hat_reshaped = C_hat.view(batch_size, BLOCKS, HAMMING_N) # [B, 64, 7]
        
        M_corrected = []
        
        # NOTE: Full error correction is complex. For PyTorch training, 
        # we simplify this by performing a hard decision (0 or 1) and extracting 
        # the data bits directly. The BCH/Hamming loss is usually computed separately 
        # or the ECC correction is used only for final BER reporting.
        
        # For training purposes, we only apply a hard decision and extract the data bits.
        # The true BCH/Hamming decoding is done offline.
        
        # In the simplest form, the data bits are the first 4 bits of the 7-bit block.
        # c = [d1, d2, d3, d4, p1, p2, p3]
        
        # Hard decision on the received codeword (probabilities to 0 or 1)
        C_hat_hard = (C_hat_reshaped > 0.5).float()
        
        # Extract the 4 data bits (d1, d2, d3, d4) from each block
        M_hat = C_hat_hard[:, :, :BLOCK_SIZE].reshape(batch_size, SOURCE_BITS) # [B, 256]
        
        return M_hat # Return the corrected/decoded data bits