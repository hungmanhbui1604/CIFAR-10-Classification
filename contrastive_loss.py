import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, mode='scl', temperature=0.1):
        super().__init__()

        self.mode = mode
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, projections, labels):
        device = projections.device
        batch_size = labels.shape[0]
        views = projections.shape[0] // batch_size
        projections = F.normalize(projections, dim=-1)

        if self.mode == 'scl':
            labels = labels.contiguous().view(-1, 1).repeat(views, 1)
        else:
            labels = torch.arange(batch_size, device=device).view(-1, 1).repeat(views, 1)

        label_mask = torch.eq(labels, labels.T).float().to(device)
        anchor_mask = ~torch.eye(batch_size * views, dtype=torch.bool, device=device)
        positive_mask = label_mask * anchor_mask

        similarities = torch.matmul(projections, projections.T)
        logits = similarities / self.temperature

        logit_max, _ = torch.max(logits, dim=1, keepdim=True)
        stable_logits = logits - logit_max.detach()

        exp_logits = torch.exp(stable_logits) * anchor_mask
        log_probs = stable_logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        positives = positive_mask.sum(1).clamp(min=self.eps)
        mean_positive_log_prob = (positive_mask * log_probs).sum(1) / positives

        loss = -mean_positive_log_prob.mean()
        return loss