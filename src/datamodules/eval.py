from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from .build import build_dataloaders
from ..models.resnet18 import Network


REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_CKPT = CKPT_DIR / "best.pth"
DEFAULT_VIZ_DIR = REPO_ROOT / "runs" / "eval_viz"


def _to_pixel_coords(landmarks: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    # landmarks: (B, 68, 2) in normalized [-0.5, 0.5] space
    h, w = image.shape[-2], image.shape[-1]
    scale = torch.tensor([w, h], device=landmarks.device, dtype=landmarks.dtype)
    return (landmarks + 0.5) * scale


def _inter_ocular_distance(gt_pixels: torch.Tensor) -> torch.Tensor:
    # 68-point markup: left eye 36-41, right eye 42-47
    left_eye = gt_pixels[:, 36:42].mean(dim=1)
    right_eye = gt_pixels[:, 42:48].mean(dim=1)
    return torch.norm(left_eye - right_eye, dim=1)


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> dict:
    model.eval()
    errors = []
    inter_ocular = []

    with torch.no_grad():
        for images, landmarks in dataloader:
            images = images.to(device)
            landmarks = landmarks.to(device)

            preds = model(images).view(-1, 68, 2)

            gt_pixels = _to_pixel_coords(landmarks, images)
            pred_pixels = _to_pixel_coords(preds, images)

            per_point_error = torch.norm(pred_pixels - gt_pixels, dim=2)  # (B, 68)
            errors.append(per_point_error)
            inter_ocular.append(_inter_ocular_distance(gt_pixels))

    errors = torch.cat(errors, dim=0)  # (N, 68)
    inter_ocular = torch.cat(inter_ocular, dim=0)  # (N,)

    mean_point_error = errors.mean(dim=1)  # (N,)
    inter_ocular = torch.clamp(inter_ocular, min=1e-6)
    nme = (mean_point_error / inter_ocular).cpu().numpy()

    rmse = torch.sqrt((errors ** 2).mean()).item()
    mae = errors.mean().item()

    # CED / AUC up to 0.08 (common in 300W)
    ced_threshold = 0.08
    failure_rate = float((nme > ced_threshold).mean())
    nme_sorted = np.sort(nme)
    ced = np.arange(1, len(nme_sorted) + 1) / len(nme_sorted)
    mask = nme_sorted <= ced_threshold
    if mask.any():
        x = nme_sorted[mask]
        y = ced[mask]
        if hasattr(np, "trapz"):
            area = np.trapz(y, x)
        elif hasattr(np, "trapezoid"):
            area = np.trapezoid(y, x)
        else:
            area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)
        auc = float(area / ced_threshold)
    else:
        auc = 0.0

    return {
        "rmse_px": rmse,
        "mae_px": mae,
        "nme_mean": float(nme.mean()),
        "nme_median": float(np.median(nme)),
        "auc_0.08": auc,
        "failure_rate_0.08": failure_rate,
    }


def visualize_samples(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    count: int,
    out_dir: Path,
) -> None:
    if count <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    images, landmarks = next(iter(dataloader))
    images = images.to(device)
    landmarks = landmarks.to(device)

    with torch.no_grad():
        preds = model(images).view(-1, 68, 2)

    gt_pixels = _to_pixel_coords(landmarks, images)
    pred_pixels = _to_pixel_coords(preds, images)

    plt.figure(figsize=(10, 2.5 * count))
    for img_num in range(min(count, images.shape[0])):
        plt.subplot(count, 1, img_num + 1)
        plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap="gray")
        plt.scatter(pred_pixels[img_num, :, 0].cpu(), pred_pixels[img_num, :, 1].cpu(), c="r", s=8)
        plt.scatter(gt_pixels[img_num, :, 0].cpu(), gt_pixels[img_num, :, 1].cpu(), c="g", s=8)
        plt.axis("off")
    plt.tight_layout()
    outfile = out_dir / "eval_samples.png"
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 68-point face landmark model.")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--viz", type=int, default=8, help="number of samples to visualize")
    parser.add_argument("--viz-dir", type=str, default=str(DEFAULT_VIZ_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, valid_loader = build_dataloaders(
        root_dir="data/",
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = Network().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    metrics = evaluate(model, valid_loader, device)
    print("Evaluation metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    visualize_samples(model, valid_loader, device, args.viz, Path(args.viz_dir))


if __name__ == "__main__":
    main()
