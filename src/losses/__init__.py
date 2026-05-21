"""Loss functions for the DL trainer.

The default training objective is **weighted Pearson**, which directly
matches the competition metric. ``WeightedMSE`` is kept around as a
warm-up loss and ``CompositeLoss`` lets you blend the two.

Each loss returns a *minimisation* objective:

  * weighted Pearson -> ``1 - corr``  (higher correlation = lower loss)
  * weighted MSE     -> the weighted mean-squared error
  * composite        -> ``alpha * weighted_mse + (1 - alpha) * (1 - corr)``

All losses respect a per-element mask (True = include in the loss). Use
the mask to exclude warm-up steps from the gradient.
"""

from losses.pearson import WeightedPearsonLoss, weighted_pearson_loss
from losses.mse import WeightedMSELoss, weighted_mse_loss
from losses.composite import CompositeLoss

LOSS_REGISTRY = {
    "weighted_pearson": WeightedPearsonLoss,
    "weighted_mse": WeightedMSELoss,
    "composite": CompositeLoss,
}


def build_loss(name: str, **kwargs):
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss: {name!r}. Available: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


__all__ = [
    "WeightedPearsonLoss",
    "WeightedMSELoss",
    "CompositeLoss",
    "weighted_pearson_loss",
    "weighted_mse_loss",
    "build_loss",
    "LOSS_REGISTRY",
]
