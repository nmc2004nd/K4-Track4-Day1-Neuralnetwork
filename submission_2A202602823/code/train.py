"""train.py — Pipeline huấn luyện, đánh giá và xuất dự đoán.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). lr=0.1 được chọn bằng validation
# sau sweep ngắn {0.01, 0.03, 0.1}; tập eval không tham gia lựa chọn.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.1,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm)
    if cm.shape != (7, 7):
        raise ValueError(f"cm phải có shape (7, 7), nhận được {cm.shape}")
    if np.any(cm < 0):
        raise ValueError("cm không được chứa giá trị âm")

    true_positive = np.diag(cm).astype(np.float64)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(7, dtype=np.float64),
        where=denominator != 0,
    )
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    model.eval()
    predictions = [
        model(X[start:start + batch_size]).argmax(dim=1)
        for start in range(0, len(X), batch_size)
    ]
    if not predictions:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(predictions)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if loss_name not in {"ce", "mse"}:
        raise ValueError("loss_name phải là 'ce' hoặc 'mse'")
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X và y phải cùng độ dài và không được rỗng")

    model.eval()
    total_loss = 0.0
    confusion = np.zeros((7, 7), dtype=np.int64)

    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        else:
            targets = F.one_hot(yb, num_classes=7).to(dtype=logits.dtype)
            total_loss += F.mse_loss(logits, targets, reduction="sum").item()

        pred = logits.argmax(dim=1)
        encoded = (yb.to(torch.int64) * 7 + pred.to(torch.int64)).detach().cpu()
        confusion += torch.bincount(encoded, minlength=49).reshape(7, 7).numpy()

    denominator = len(X) if loss_name == "ce" else len(X) * 7
    return {
        "loss": float(total_loss / denominator),
        "acc": float(np.trace(confusion) / len(X)),
        "macro_f1": macro_f1_from_confusion(confusion),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        targets = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        # PyTorch lấy trung bình trên mọi phần tử B x C và không có hệ số 1/2.
        return F.mse_loss(logits, targets, reduction="mean")
    raise ValueError("loss_name phải là 'ce' hoặc 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError(f"Kiến trúc {hidden} không nằm trong EXPECTED_PARAMS")
    if cfg["lr"] is None or cfg["lr"] <= 0:
        raise ValueError("Cần đặt cfg['lr'] là một số dương trước khi huấn luyện")
    if cfg["batch"] <= 0 or cfg["epochs"] <= 0:
        raise ValueError("batch và epochs phải là số nguyên dương")

    required_data = ("X_tr", "y_tr", "X_val", "y_val")
    missing = [key for key in required_data if key not in data]
    if missing:
        raise KeyError(f"data thiếu các khoá: {missing}")
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if any(tensor.device != device for tensor in (y_tr, X_val, y_val)):
        raise ValueError("Mọi tensor train/val phải nằm trên cùng một device")

    precision = cfg["precision"].lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là 'fp32', 'fp16' hoặc 'bf16'")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("Huấn luyện FP16 với GradScaler yêu cầu CUDA")
    if precision == "bf16" and device.type not in {"cpu", "cuda"}:
        raise ValueError("Pipeline BF16 hiện hỗ trợ CPU hoặc CUDA")

    set_seed(int(cfg["seed"]))
    model = MLP(
        hidden=hidden,
        dropout=float(cfg["dropout"]),
        init=cfg["init"],
        in_features=X_tr.shape[1],
        num_classes=7,
    ).to(device)
    actual_params = count_params(model)
    assert actual_params == EXPECTED_PARAMS[hidden], (
        f"Sai số tham số cho {hidden}: {actual_params} != {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
        betas=tuple(cfg.get("betas", (0.9, 0.999))),
        eps=float(cfg.get("eps", 1e-8)),
    )
    scheduler = build_scheduler(
        optimizer,
        cfg.get("scheduler"),
        total_steps=int(cfg["epochs"]),
        **cfg.get("scheduler_kwargs", {}),
    )

    use_scaler = precision == "fp16"
    if use_scaler:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            except TypeError:  # tương thích một số bản PyTorch 2.x cũ hơn
                scaler = torch.amp.GradScaler(enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision)
    use_autocast = precision != "fp32"

    try:
        generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    except (RuntimeError, TypeError):
        # Một số backend không cung cấp Generator riêng; global seed vẫn đã được đặt.
        generator = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step0 = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    diverged = False

    def synchronize_device():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        synchronize_device()
        epoch_start = time.perf_counter()
        model.train()
        grad_norms = []

        for xb, yb in iterate_batches(
            X_tr, y_tr, int(cfg["batch"]), generator=generator, shuffle=True
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if not torch.isfinite(loss).item():
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                # Gradient norm phải được đo trên gradient thật, không phải gradient đã scale.
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            grad_norm = clip_gradients(model.parameters(), cfg.get("clip_norm"))
            if not math.isfinite(grad_norm):
                diverged = True
                optimizer.zero_grad(set_to_none=True)
                break
            grad_norms.append(grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        train_metrics = evaluate(model, X_tr, y_tr, loss_name=cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
        if not math.isfinite(train_metrics["loss"]) or not math.isfinite(val_metrics["loss"]):
            diverged = True
        synchronize_device()
        elapsed = time.perf_counter() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(grad_norms)) if grad_norms else float("nan"))
        history["epoch_time_s"].append(elapsed)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if cfg.get("verbose", True):
            print(
                f"[{cfg['exp_id']}] epoch {epoch:02d}/{cfg['epochs']}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}, "
                f"val_f1={val_metrics['macro_f1']:.4f}, "
                f"grad_norm={history['grad_norm'][-1]:.4f}, time={elapsed:.2f}s"
            )

        if scheduler is not None:
            scheduler.step()
        if diverged:
            break

    if best_state is None:
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        best_epoch = len(history["epoch"])
        best_val_loss = history["val_loss"][-1]

    best_index = best_epoch - 1
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "step0_loss": step0["loss"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "val_acc": history["val_acc"][best_index],
        "val_macro_f1": history["val_macro_f1"][best_index],
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])),
        "peak_mem_MB": float(peak_mem_mb),
        "diverged": diverged,
    }
    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds):
        raise ValueError("row_id và preds phải là mảng 1D có cùng độ dài")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("Mỗi row_id phải xuất hiện đúng một lần")
    if not np.issubdtype(row_id.dtype, np.integer):
        raise ValueError("row_id phải có kiểu số nguyên")
    if not np.issubdtype(preds.dtype, np.integer):
        raise ValueError("preds phải có kiểu số nguyên")
    if np.any((preds < 0) | (preds > 6)):
        raise ValueError("preds chỉ được chứa nhãn từ 0 đến 6")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "pred"])
        writer.writerows(zip(row_id.tolist(), preds.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    merged_cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = MLP(
        hidden=tuple(merged_cfg["hidden"]),
        dropout=float(merged_cfg["dropout"]),
        init=merged_cfg["init"],
        in_features=data["X_eval"].shape[1],
        num_classes=7,
    )
    model.load_state_dict(result["best_state"])
    model.to(device)
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
