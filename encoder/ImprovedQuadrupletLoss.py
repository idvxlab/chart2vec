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
    """
        Multitasking loss function: Linear interpolation loss + Triplet loss
        Inputs:
            `rep_x1`: Vector of the first chart in the quaternion represented via chart2ve.
            `rep_x2`: Vector of the second chart in the quaternion represented via chart2vec.
            `rep_x3`: Vector of the third chart in the quaternion represented via chart2vec.
            `rep_y`: Vector of the fourth chart in the quaternion represented via chart2vec.
    """
    def __init__(self, distance_metric=DistanceMetric.EUCLIDEAN, margin: float = 5):
        super(ImprovedQuadrupletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, rep_x1, rep_x2, rep_x3, rep_y):

        m = (rep_x1+rep_x3)/2
        coefficient = 0.25

        # -------------Linear interpolation loss-------------
        # difference between the computed and original intermediate vectors
        dist_linear = self.distance_metric(rep_x2, m)
        # calculate the distance between two of the first three charts
        distance_pos12 = self.distance_metric(rep_x1, rep_x2)
        distance_pos23 = self.distance_metric(rep_x2, rep_x3)
        distance_pos13 = self.distance_metric(rep_x1, rep_x3)
        dist_diff = distance_pos12 + distance_pos23 + distance_pos13
        # sum of the two terms
        linear_losses = dist_linear + coefficient*dist_diff
        
        # -------------Triplet loss-------------
        distance_neg = self.distance_metric(rep_x1, rep_y)  
        triplet_loss = F.relu(distance_pos13-distance_neg+self.margin)

        # -------------total loss-------------
        losses = coefficient*linear_losses + triplet_loss
        return losses.mean()
