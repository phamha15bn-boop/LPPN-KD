# Derived from the GPL-3.0 csnbench LPPNM implementation.
# LPPN authors: Ziye Yu and Weitao Wang.
# Upstream source: https://github.com/cangyeone/csnbench/blob/66614c5c1ffc13fcabd0f4c0b7a599e17bebb839/models/LPPNM.py
# Modified by Qineng Song in 2026 (see THIRD_PARTY_NOTICES.md).
# Changes: extracted and adapted the supervised loss for LPPN-KD.
# See LICENSE and THIRD_PARTY_NOTICES.md for attribution and terms.

"""Original supervised LPPN classification and arrival-offset loss."""

import torch.nn as nn


class LPPNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    def forward(self, prediction, label):
        predicted_class, predicted_time = prediction
        target_class = label[:, 0, :].long()
        target_time = label[:, 1, :].float()
        classification_loss = self.ce(predicted_class, target_class)
        masked_time_loss = self.mse(predicted_time, target_time) * target_class.clamp(
            0, 1
        ).float()
        return classification_loss + masked_time_loss.sum() / 10
