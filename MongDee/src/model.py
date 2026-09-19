"""Multi-task Age/Gender/Race model: a shared ResNet backbone (ImageNet
pretrained initialization, never a pretrained FairFace model — every weight
below is actually trained on this project's dataset/) feeding three
independent classification heads.

Head order (age, gender, race) is arbitrary for this class's own forward()
dict, but export_combined_state_dict() below concatenates them in
[race, gender, age] order specifically to match
core/attributes.py's FairFaceBackend, which expects a single
Linear(in_features, 7 + 2 + 9) fc layer laid out that way (the official
joojs/fairface res34_fair_align_multi_7_*.pt checkpoint's own layout) — so
a model trained here can be dropped straight into the existing booth app
via --fairface-checkpoint without any conversion step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_BACKBONES = {"resnet18": 512, "resnet34": 512}


def _build_backbone(architecture: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision import models

    if architecture == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif architecture == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r} (expected resnet18 or resnet34)")
    in_features = net.fc.in_features
    net.fc = nn.Identity()  # heads below replace it; backbone just returns the pooled feature vector
    return net, in_features


class MultiTaskFaceModel(nn.Module):
    def __init__(self, num_age: int, num_gender: int, num_race: int,
                 architecture: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        self.num_age = num_age
        self.num_gender = num_gender
        self.num_race = num_race
        self.backbone, feat_dim = _build_backbone(architecture, pretrained)
        self.age_head = nn.Linear(feat_dim, num_age)
        self.gender_head = nn.Linear(feat_dim, num_gender)
        self.race_head = nn.Linear(feat_dim, num_race)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "age": self.age_head(features),
            "gender": self.gender_head(features),
            "race": self.race_head(features),
        }

    def export_combined_state_dict(self) -> dict:
        """A plain resnet* state_dict with a single fc layer of width
        (num_race + num_gender + num_age), weights copied from this model's
        three heads concatenated in [race, gender, age] order — the exact
        shape core/attributes.py's FairFaceBackend loads via
        `model.load_state_dict(torch.load(checkpoint_path))`. Only meaningful
        when num_race == 7 and num_gender == 2 (FairFaceBackend's
        NUM_RACE_CLASSES / GENDER_LABELS are fixed at those sizes); a model
        trained with a different race/gender class count still trains and
        saves fine, it just isn't drop-in compatible with that specific
        consumer.
        """
        from torchvision import models

        builder = models.resnet18 if self.architecture == "resnet18" else models.resnet34
        combined = builder(weights=None)
        combined.fc = nn.Linear(combined.fc.in_features, self.num_race + self.num_gender + self.num_age)

        own_backbone_sd = self.backbone.state_dict()
        combined_sd = combined.state_dict()
        for key in combined_sd:
            if key.startswith("fc."):
                continue
            combined_sd[key] = own_backbone_sd[key]

        fc_weight = torch.cat([self.race_head.weight, self.gender_head.weight, self.age_head.weight], dim=0)
        fc_bias = torch.cat([self.race_head.bias, self.gender_head.bias, self.age_head.bias], dim=0)
        combined_sd["fc.weight"] = fc_weight
        combined_sd["fc.bias"] = fc_bias
        return combined_sd


def build_model(architecture: str, num_age: int, num_gender: int, num_race: int,
                 pretrained: bool = True) -> MultiTaskFaceModel:
    return MultiTaskFaceModel(num_age=num_age, num_gender=num_gender, num_race=num_race,
                               architecture=architecture, pretrained=pretrained)
