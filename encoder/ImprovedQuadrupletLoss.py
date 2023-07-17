from torch import nn
from enum import Enum
import torch.nn.functional as F


class DistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    def COSINE(x, y): return 1 - F.cosine_similarity(x, y)
    def EUCLIDEAN(x, y): return F.pairwise_distance(x, y, p=2)
    def MANHATTAN(x, y): return F.pairwise_distance(x, y, p=1)


class ImprovedQuadrupletLoss(nn.Module):
    def __init__(self, distance_metric=DistanceMetric.EUCLIDEAN, margin: float = 5):
        super(ImprovedQuadrupletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, rep_x1, rep_x2, rep_x3, rep_y):
        m = (rep_x1+rep_x3)/2
        coefficient = 0.25

        distance_pos12 = self.distance_metric(rep_x1, rep_x2)
        distance_pos23 = self.distance_metric(rep_x2, rep_x3)
        distance_pos13 = self.distance_metric(rep_x1, rep_x3)
        distance_neg = self.distance_metric(rep_x1, rep_y)
        dist_linear = self.distance_metric(rep_x2, m)
        dist_diff = distance_pos12 + distance_pos23 + distance_pos13

        linear_losses = dist_linear + coefficient*dist_diff
        triplet_loss = F.relu(distance_pos13-distance_neg+self.margin)

        losses = coefficient*linear_losses + triplet_loss
        return losses.mean()
