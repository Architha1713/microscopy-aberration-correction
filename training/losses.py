import torch
import torch.nn as nn
import torch.nn.functional as F

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def edge_loss(prediction, target):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=prediction.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=prediction.device).view(1, 1, 3, 3)
    pred_edge_x = F.conv2d(prediction, sobel_x, padding=1)
    pred_edge_y = F.conv2d(prediction, sobel_y, padding=1)
    target_edge_x = F.conv2d(target, sobel_x, padding=1)
    target_edge_y = F.conv2d(target, sobel_y, padding=1)
    pred_edges = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
    target_edges = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
    return F.l1_loss(pred_edges, target_edges)

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.5, ssim_weight=0.3, edge_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)
        ssim_val = ssim(prediction, target)
        ssim_loss = 1 - ssim_val
        edge_l = edge_loss(prediction, target)
        total = (self.l1_weight * l1_loss) + (self.ssim_weight * ssim_loss) + (self.edge_weight * edge_l)
        return total, l1_loss.item(), ssim_val.item()