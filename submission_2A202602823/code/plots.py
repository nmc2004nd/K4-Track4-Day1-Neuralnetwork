"""plots.py — Biểu đồ cho từng thí nghiệm và so sánh nhiều cấu hình.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg = result["cfg"]
    history = result["history"]
    summary = result["summary"]
    epochs = history["epoch"]
    if not epochs:
        raise ValueError("Không thể vẽ một lịch sử rỗng")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", markersize=3, label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", markersize=3, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], marker="o", markersize=3, label="accuracy")
    axes[1].plot(
        epochs, history["val_macro_f1"], marker="o", markersize=3, label="macro-F1"
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(epochs, history["grad_norm"], marker="o", markersize=3, label="grad norm")
    axes[2].set_title("Gradient norm (before clipping)")
    axes[2].set_ylabel("Global L2 norm")
    axes[2].legend()

    best_epoch = summary.get("best_epoch")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if best_epoch is not None:
            axis.axvline(best_epoch, color="tab:red", linestyle="--", alpha=0.65,
                         label="best epoch")
        axis.legend()

    config_text = (
        f"optimizer={cfg.get('optimizer')}, lr={cfg.get('lr')}, batch={cfg.get('batch')}, "
        f"init={cfg.get('init')}, dropout={cfg.get('dropout')}, "
        f"precision={cfg.get('precision')}"
    )
    fig.suptitle(f"{cfg.get('exp_id', 'experiment')} — {config_text}", fontsize=11)
    fig.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results không được rỗng")

    fig, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise KeyError(f"Metric {metric!r} không có trong history")
        axis.plot(
            history["epoch"],
            history[metric],
            marker="o",
            markersize=3,
            label=result["cfg"].get("exp_id", "experiment"),
        )

    axis.set_title(title or f"Comparison — {metric}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
