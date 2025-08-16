import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, mode, temperature):
        super().__init__()

        self.mode = mode
        self.temperature = temperature
        self.epsilon = 1e-7

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
        log_probs = stable_logits - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        positives = positive_mask.sum(1).clamp(min=self.epsilon)
        mean_positive_log_prob = (positive_mask * log_probs).sum(1) / positives

        loss = -mean_positive_log_prob.mean()
        return loss

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, mode, temperature):
        super().__init__()
        self.mode = mode
        self.temperature = temperature
        self.epsilon = 1e-7

    def forward(self, projections, labels):
        device = labels.device
        hcl_batch_size, hard_negative_batch_size = labels.shape
        projections = F.normalize(projections, dim=-1)

        if self.mode == 'scl':
            labels = labels.contiguous()
            label_mask = (labels == labels[:, 0].view(-1, 1)).float()
            anchor_mask = torch.ones_like(label_mask, device=device)
            anchor_mask[:, 0] = 0
            positive_mask = label_mask * anchor_mask
        elif self.mode == 'simclr':
            anchor_mask = torch.ones(hcl_batch_size, hard_negative_batch_size, device=device)
            anchor_mask[:, 0] = 0
            positive_mask = torch.zeros(hcl_batch_size, hard_negative_batch_size, device=device)
            positive_mask[:, 1] = 1.

        anchors = projections[:, 0].unsqueeze(1)
        similarities = torch.sum(anchors*projections, dim=-1)
        logits = similarities / self.temperature

        logit_max, _ = torch.max(logits, dim=1, keepdim=True)
        stable_logits = logits - logit_max.detach()

        exp_logits = torch.exp(stable_logits) * anchor_mask
        log_probs = stable_logits - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        positives = positive_mask.sum(1).clamp(min=self.epsilon)
        mean_positive_log_prob = (positive_mask * log_probs).sum(1) / positives

        loss = -mean_positive_log_prob.mean()
        return loss