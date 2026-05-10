"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ─────────────────────────────────────────────
# TODO: Implement the NoamScheduler class below
# ─────────────────────────────────────────────

class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    Applies a warm-up phase where LR increases linearly, followed by
    a decay phase where LR decreases proportional to the inverse square
    root of the step number.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        d_model          (int)  : Model dimensionality (embedding size).
        warmup_steps     (int)  : Number of warm-up steps before decay begins.
        last_epoch       (int)  : The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        # Must be stored before super().__init__() because the parent calls step() → get_lr()
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    # ------------------------------------------------------------------
    def _get_lr_scale(self) -> float:
        """
        Compute the Noam scaling factor for the current step.

        Returns:
            float: The scalar multiplier applied to the base learning rate.
        """
        step = self.last_epoch + 1   # avoid step=0 division
        return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    # ------------------------------------------------------------------
    def get_lr(self) -> list[float]:
        """
        Compute learning rates for every param group.

        Called internally by PyTorch's scheduler machinery each step.

        Returns:
            list[float]: New learning rate for each param group in the optimizer.
        """
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
# Helper — do NOT modify
# ──────────────────────────────────────────────────────────────────────

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.

    Args:
        d_model      (int): Model dimensionality.
        warmup_steps (int): Warm-up steps.
        total_steps  (int): Number of steps to simulate.

    Returns:
        list[float]: LR value at each step (length == total_steps).
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ──────────────────────────────────────────────────────────────────────
# Quick visual check — run:  python noam_lr_scheduler.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
