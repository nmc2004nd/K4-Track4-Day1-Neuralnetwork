"""data.py — Tiện ích nạp, chia, chuẩn hoá và tạo batch dữ liệu.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_dir = Path(processed_dir)
    train_path = processed_dir / "train.npz"
    eval_path = processed_dir / "eval.npz"

    with np.load(train_path) as train_data:
        X_train_full = train_data["X"]
        y_train_full = train_data["y"]
    with np.load(eval_path) as eval_data:
        X_eval = eval_data["X"]
        y_eval = eval_data["y"]
        eval_row_id = eval_data["row_id"]

    def _check_xy(X, y, split_name):
        assert X.ndim == 2 and X.shape[1] == 54, \
            f"X_{split_name} phải có shape (N, 54), nhận được {X.shape}"
        assert y.ndim == 1 and len(y) == len(X), \
            f"y_{split_name} phải có shape ({len(X)},), nhận được {y.shape}"
        assert X.dtype == np.float32, \
            f"X_{split_name} phải có dtype float32, nhận được {X.dtype}"
        assert y.dtype == np.int64, \
            f"y_{split_name} phải có dtype int64, nhận được {y.dtype}"
        assert np.all((0 <= y) & (y <= 6)), \
            f"y_{split_name} phải chỉ chứa nhãn từ 0 đến 6"

    _check_xy(X_train_full, y_train_full, "train")
    _check_xy(X_eval, y_eval, "eval")
    assert eval_row_id.ndim == 1 and len(eval_row_id) == len(X_eval), \
        f"eval_row_id phải có shape ({len(X_eval)},), nhận được {eval_row_id.shape}"
    assert eval_row_id.dtype == np.int64, \
        f"eval_row_id phải có dtype int64, nhận được {eval_row_id.dtype}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm trong khoảng (0, 1)")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr phải có ít nhất {N_NUMERIC} cột")
    if len(X_tr) == 0:
        raise ValueError("Không thể fit standardizer trên tập train rỗng")

    # Tính thống kê ở float64 để tránh sai số tích luỹ trên tập dữ liệu lớn.
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64, copy=False)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X phải có ít nhất {N_NUMERIC} cột")

    mean = np.asarray(mean)
    std = np.asarray(std)
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError(
            f"mean và std phải có shape ({N_NUMERIC},), "
            f"nhận được {mean.shape} và {std.shape}"
        )

    # Cột có độ lệch chuẩn bằng 0 là hằng số; chia cho 1 sẽ biến nó thành 0.
    safe_std = np.where(std == 0, 1, std)
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / safe_std
    return X_scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    data = {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
    }

    majority_class = torch.bincount(data["y_tr"], minlength=7).argmax()
    majority_val_accuracy = (data["y_val"] == majority_class).float().mean().item()
    print(
        f"train: {tuple(data['X_tr'].shape)}, "
        f"val: {tuple(data['X_val'].shape)}, "
        f"eval: {tuple(data['X_eval'].shape)}"
    )
    print(
        f"Majority-class baseline trên val "
        f"(lớp {majority_class.item()}): {majority_val_accuracy:.4f}"
    )
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")

    if shuffle:
        indices = torch.randperm(len(X), generator=generator, device=X.device)
    else:
        indices = torch.arange(len(X), device=X.device)

    # Giữ lại batch cuối dù có ít hơn batch_size mẫu để không bỏ dữ liệu.
    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]
