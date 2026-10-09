"""Create a controlled visual comparison of four convolution operators.

The figure uses one MNIST image and a 3x3 averaging kernel.  Every spatial
kernel coefficient equals 1/9.  The deformable panel uses a fixed, slightly
expanded sampling grid only to illustrate the operator; trained DCNv1 offsets
are input-dependent and are visualised separately in the experiment outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "mnist"
OUT = ROOT / "assets" / "operator_demo_mnist.png"


def deformable_average(x: torch.Tensor) -> torch.Tensor:
    """Average nine bilinearly sampled locations with a fixed offset grid."""
    _, _, height, width = x.shape
    base_y, base_x = torch.meshgrid(
        torch.arange(height, dtype=x.dtype),
        torch.arange(width, dtype=x.dtype),
        indexing="ij",
    )
    # A symmetric expansion: corners move away from the centre by 0.55 pixels.
    offsets = [
        (-0.55, -0.55), (-0.25, 0.0), (-0.55, 0.55),
        (0.0, -0.25), (0.0, 0.0), (0.0, 0.25),
        (0.55, -0.55), (0.25, 0.0), (0.55, 0.55),
    ]
    samples = []
    for ky, kx in offsets:
        yy = base_y + ky
        xx = base_x + kx
        # grid_sample expects coordinates in [-1, 1], ordered as (x, y).
        grid = torch.stack(
            (2 * xx / (width - 1) - 1, 2 * yy / (height - 1) - 1), dim=-1
        ).unsqueeze(0)
        samples.append(F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True))
    return torch.stack(samples).mean(0)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dataset = MNIST(DATA_ROOT, train=False, download=True)
    image, label = dataset[1]  # A clear handwritten "2".
    x = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None] / 255.0
    kernel = torch.full((1, 1, 3, 3), 1 / 9)

    standard = F.conv2d(x, kernel, padding=1)
    dilated = F.conv2d(x, kernel, padding=2, dilation=2)
    depthwise = F.conv2d(x, kernel, padding=1, groups=1)
    pointwise = F.conv2d(depthwise, torch.ones((1, 1, 1, 1)))
    deformable = deformable_average(x)

    panels = [
        (x, "Вход"),
        (standard, "Обычная 3×3"),
        (dilated, "Dilated 3×3, d=2"),
        (pointwise, "Depthwise 3×3 + 1×1"),
        (deformable, "Deformable 3×3"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.5, 3.2), constrained_layout=True)
    for ax, (tensor, title) in zip(axes, panels):
        ax.imshow(tensor[0, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"MNIST: цифра {label}. Все 9 пространственных весов равны 1/9", fontsize=17, y=1.06)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
