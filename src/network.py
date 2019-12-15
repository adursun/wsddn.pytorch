import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet
from torchvision.ops import roi_pool


class WSDDN(nn.Module):
    base = alexnet(pretrained=False)

    def __init__(self):
        super().__init__()
        self.base.load_state_dict(torch.load("../states/alexnet-owt-4df8aa71.pth"))
        self.features = self.base.features[:-1]
        self.fcs = self.base.classifier[1:-1]
        self.fc_c = nn.Linear(4096, 20)
        self.fc_d = nn.Linear(4096, 20)

    def forward(self, batch_imgs, batch_boxes, batch_scores):
        # assume batch size is 1
        batch_boxes = [batch_boxes[0]]

        out = self.features(batch_imgs)  # [1, 256, 21, 29]
        out = roi_pool(out, batch_boxes, (6, 6), 1.0 / 16)  # [4000, 256, 6, 6]
        out = out.view(-1, 9216)  # [4000, 9216]
        out = out * batch_scores[0]  # apply box scores
        out = self.fcs(out)  # [4000, 4096]

        classification_scores = F.softmax(self.fc_c(out), dim=1)
        detection_scores = F.softmax(self.fc_d(out), dim=0)
        combined_scores = classification_scores * detection_scores
        return combined_scores

    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss
