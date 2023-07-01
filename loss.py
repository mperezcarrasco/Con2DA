import torch
import numpy as np

from torch.nn import functional as F


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
    def negative_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask)
        return mask.to(self.device).type(torch.bool)

    def forward(self, zw, zs):
        representation = torch.cat([zw, zs], dim=0)
        
        # (2*batch_size, 2*batch_size)
        similarity_matrix = F.cosine_similarity(representation.unsqueeze(1), representation.unsqueeze(0), dim=-1)
        
        # filtering positive scores between z_weak,z_strong
        ws_pos = torch.diag(similarity_matrix, self.batch_size)
        
        # filtering positive scores between z_strong,z_weak
        sw_pos = torch.diag(similarity_matrix, -self.batch_size)
        
        positives = torch.cat([ws_pos, sw_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.negative_mask()].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = F.cross_entropy(logits, labels, reduction="sum")

        return loss / (2 * self.batch_size)
