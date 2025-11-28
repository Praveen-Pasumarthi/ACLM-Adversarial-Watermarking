import torch
import torch.nn.functional as F
import numpy as np

HAMMING_N = 7
HAMMING_K = 4
HAMMING_R = 3
BLOCK_SIZE = HAMMING_K
SOURCE_BITS = 256
BLOCKS = SOURCE_BITS // HAMMING_K
CODEWORD_BITS = BLOCKS * HAMMING_N

class Hamming74:
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def _encode_single_block(self, d):
        d = d.long().tolist() # Convert to list of 0s and 1s
        d1, d2, d3, d4 = d[0], d[1], d[2], d[3]
        
        # Calculate parity bits (Even Parity)
        p1 = d1 ^ d2 ^ d4 
        p2 = d1 ^ d3 ^ d4 
        p3 = d2 ^ d3 ^ d4 
        
        # Codeword c = [d1, d2, d3, d4, p1, p2, p3]
        return torch.tensor([d1, d2, d3, d4, p1, p2, p3], dtype=torch.float32, device=self.device)

    def encode(self, M):
        batch_size = M.size(0)
        M_reshaped = M.view(batch_size, BLOCKS, BLOCK_SIZE) # [B, 64, 4]
        
        C = []
        for i in range(batch_size):
            codeword_batch = []
            for j in range(BLOCKS):
                codeword_batch.append(self._encode_single_block(M_reshaped[i, j, :]))
            C.append(torch.cat(codeword_batch))
            
        return torch.stack(C)

    def decode_and_correct(self, C_hat):
        batch_size = C_hat.size(0)
        C_hat_reshaped = C_hat.view(batch_size, BLOCKS, HAMMING_N)
        
        M_corrected = []
        C_hat_hard = (C_hat_reshaped > 0.5).float()
        M_hat = C_hat_hard[:, :, :BLOCK_SIZE].reshape(batch_size, SOURCE_BITS)
        
        return M_hat
    
def calculate_ber(M, M_hat):
    incorrect_bits = torch.sum(torch.abs(M_hat - M)).item()
    ber = incorrect_bits / (M.size(0) * M.size(1))
    return ber