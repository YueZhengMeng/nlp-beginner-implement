import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, weight_device=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        if self.weight is not None:
            self.weight = torch.as_tensor(self.weight, dtype=torch.float32)
        if weight_device is not None:
            self.weight = self.weight.to(weight_device)
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.weight, reduction=reduction)

    def forward(self, logit, target):
        loss = self.cross_entropy(logit, target)
        return loss


class MultiFocalLoss(nn.Module):

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        alpha = self.alpha.to(logit.device)
        prob = torch.softmax(logit, dim=1)

        if prob.dim() > 2:
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()
            prob = prob.view(-1, prob.size(-1))

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth
        logpt = torch.log(prob)
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss
