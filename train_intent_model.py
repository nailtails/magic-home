import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 20
STEP_SIZE = 3
FEATURES = ["amp_y", "vel_y", "delta_y", "delta_x", "weight"]
RANDOM_SEED = 42


class Tee:
    """Write stdout to both console and a log file."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════
# Data loading and frame-level deduplication
# ════════════════════════════════════════════════════

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["frame", "label", "segment", "source", "active"] + FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    df = df[df["label"].isin([0, 1])].copy()
    df[FEATURES] = df[FEATURES].fillna(0.0)

    print("\n📊 数据概览")
    print(f"   CSV 文件 : {csv_path}")
    print(f"   总行数   : {len(df)}")
    print(f"   label=1  : {(df['label'] == 1).sum()} 行（有意调节）")
    print(f"   label=0  : {(df['label'] == 0).sum()} 行（无意动作）")
    print(f"   来源     : {df['source'].unique().tolist()}")
    print(f"   segment数: {df[['source', 'segment']].drop_duplicates().shape[0]}")

    rows = []
    skipped = 0

    # 每帧可能有左右手多行，所以按 source + segment + frame 去重
    for (src, seg, frame), grp in df.groupby(["source", "segment", "frame"]):
        label = int(grp["label"].iloc[0])
        active = grp[grp["active"] == 1]

        if label == 1:
            # 有意动作：只保留 active=1 的行，避免全零或无效帧污染训练
            if len(active) == 0:
                skipped += 1
                continue
            row = active.loc[active["amp_y"].abs().idxmax()]
        else:
            # 无意动作：如果有 active 行，优先取 active 行；否则取 amp_y 最大的行
            if len(active) > 0:
                row = active.loc[active["amp_y"].abs().idxmax()]
            else:
                row = grp.loc[grp["amp_y"].abs().idxmax()]

        rows.append(row)

    df_clean = pd.DataFrame(rows).reset_index(drop=True)

    print(f"\n   去重后行数: {len(df_clean)}（每帧一行，跳过 {skipped} 帧未激活的 label=1）")
    print(f"   label=1  : {(df_clean['label'] == 1).sum()} 帧")
    print(f"   label=0  : {(df_clean['label'] == 0).sum()} 帧")

    return df, df_clean, skipped


# ════════════════════════════════════════════════════
# Sliding-window construction
# ════════════════════════════════════════════════════

def make_windows(df):
    """
    按 source + segment 分组切滑动窗口。
    每个窗口继承当前 source + segment 的 label。
    """
    X_list, y_list, group_list = [], [], []
    group_meta = []
    group_id = 0

    for (src, seg), grp in df.groupby(["source", "segment"]):
        grp = grp.sort_values("frame").reset_index(drop=True)
        feat = grp[FEATURES].values.astype(np.float32)
        label = int(grp["label"].iloc[0])

        if len(feat) == 0:
            continue

        original_len = len(feat)

        # 如果某个 segment 太短，则左侧补齐
        if len(feat) < WINDOW_SIZE:
            pad = np.repeat(feat[:1], WINDOW_SIZE - len(feat), axis=0)
            feat = np.vstack([pad, feat])

        n_windows = 0
        for start in range(0, len(feat) - WINDOW_SIZE + 1, STEP_SIZE):
            X_list.append(feat[start:start + WINDOW_SIZE])
            y_list.append(float(label))
            group_list.append(group_id)
            n_windows += 1

        group_meta.append({
            "group_id": group_id,
            "source": str(src),
            "segment": str(seg),
            "label": label,
            "frames_after_dedup": int(original_len),
            "windows": int(n_windows),
        })
        group_id += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(group_list)

    print("\n📦 窗口样本")
    print(f"   窗口总数: {len(X)}")
    print(f"   正样本  : {int(y.sum())} (label=1，有意调节)")
    print(f"   负样本  : {int((1 - y).sum())} (label=0，无意动作)")

    if len(X) > 0:
        pos_ratio = float(y.mean())
        neg_ratio = 1.0 - pos_ratio
        print(f"   正样本占比: {pos_ratio:.3f}")
        print(f"   负样本占比: {neg_ratio:.3f}")

    return X, y, groups, group_meta


# ════════════════════════════════════════════════════
# Group-level split
# ════════════════════════════════════════════════════

def split_data(X, y, groups, group_meta, val_ratio=0.2):
    """
    按 group 分割训练集和验证集。
    这样同一个 source + segment 产生的窗口不会同时进入训练和验证。
    """
    unique_groups = np.unique(groups)
    np.random.shuffle(unique_groups)

    n_val = max(1, int(len(unique_groups) * val_ratio))
    val_groups = set(unique_groups[:n_val])
    train_groups = set(unique_groups[n_val:])

    train_mask = np.array([g in train_groups for g in groups])
    val_mask = np.array([g in val_groups for g in groups])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("\n✂️ 数据划分")
    print(f"   训练集窗口: {len(X_train)}")
    print(f"   验证集窗口: {len(X_val)}")
    print(f"   训练集标签: 正={int(y_train.sum())} 负={int((1 - y_train).sum())}")
    print(f"   验证集标签: 正={int(y_val.sum())} 负={int((1 - y_val).sum())}")
    print(f"   训练 groups: {len(train_groups)}")
    print(f"   验证 groups: {len(val_groups)}")

    split_meta = []
    for m in group_meta:
        group_id = m["group_id"]
        part = "val" if group_id in val_groups else "train"
        row = dict(m)
        row["split"] = part
        split_meta.append(row)

    return X_train, y_train, X_val, y_val, split_meta


# ════════════════════════════════════════════════════
# Normalisation
# ════════════════════════════════════════════════════

def normalize(X_train, X_val):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm, X_val_norm, mean, std


# ════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════

def build_model(n_features):
    import torch.nn as nn

    class IntentCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

            self.head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: [batch, window, features]
            x = x.permute(0, 2, 1)  # -> [batch, features, window]
            x = self.conv(x)
            x = x.mean(dim=-1)
            x = self.head(x).squeeze(-1)
            return x

    return IntentCNN()


# ════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision_1 = tp / max(tp + fp, 1)
    recall_1 = tp / max(tp + fn, 1)
    f1_1 = 2 * precision_1 * recall_1 / max(precision_1 + recall_1, 1e-8)

    precision_0 = tn / max(tn + fn, 1)
    recall_0 = tn / max(tn + fp, 1)
    f1_0 = 2 * precision_0 * recall_0 / max(precision_0 + recall_0, 1e-8)

    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "accuracy": acc,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "macro_f1": macro_f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": threshold,
    }


def print_metrics(metrics, prefix="VAL"):
    print(
        f"[{prefix}] "
        f"acc={metrics['accuracy']:.4f} | "
        f"f1_0={metrics['f1_0']:.4f} | "
        f"f1_1={metrics['f1_1']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(
        f"[{prefix}] Confusion Matrix: "
        f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )


# ════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════

def train(model, X_train, y_train, X_val, y_val, epochs, batch_size=32):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 使用设备: {device}")

    model = model.to(device)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 训练集平衡采样
    y_train_int = y_train.astype(np.int64)
    class_counts = np.bincount(y_train_int, minlength=2)

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError(
            f"训练集中某一类为空，无法训练二分类模型。class_counts={class_counts.tolist()}"
        )

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train_int]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("\n⚖️ 训练集平衡采样")
    print(f"   原始训练集: label=0 数量={class_counts[0]}, label=1 数量={class_counts[1]}")
    print(f"   采样权重  : label=0 weight={class_weights[0]:.6f}, label=1 weight={class_weights[1]:.6f}")
    print("   说明      : 只平衡训练 batch，验证集保持原始分布")

    train_dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCELoss()

    best_macro_f1 = -1.0
    best_state = None
    best_metrics = None
    history = []

    print("\n🚀 开始训练")
    print(
        f"{'epoch':>6}  "
        f"{'loss':>8}  "
        f"{'acc':>8}  "
        f"{'f1_0':>8}  "
        f"{'f1_1':>8}  "
        f"{'macro_f1':>10}"
    )
    print("-" * 62)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        sched.step()

        model.eval()
        with torch.no_grad():
            val_prob = model(X_v).detach().cpu().numpy()
            val_true = y_v.detach().cpu().numpy()

        metrics = compute_binary_metrics(val_true, val_prob, threshold=0.5)
        avg_loss = total_loss / max(len(loader), 1)

        history_row = {
            "epoch": ep,
            "loss": avg_loss,
            **metrics,
        }
        history.append(history_row)

        # 用 macro-F1 选模型，因为它同时考虑 label=0 和 label=1
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(
                f"{ep:>6}  "
                f"{avg_loss:>8.4f}  "
                f"{metrics['accuracy']:>8.3f}  "
                f"{metrics['f1_0']:>8.3f}  "
                f"{metrics['f1_1']:>8.3f}  "
                f"{metrics['macro_f1']:>10.3f}"
            )

    if best_state is None:
        raise RuntimeError("训练失败：没有保存到 best_state。")

    model.load_state_dict(best_state)
    model = model.to(device)

    print("\n✅ 最佳模型指标（按 macro-F1 选择）")
    print_metrics(best_metrics, prefix="BEST VAL")

    return model, best_metrics, history


# ════════════════════════════════════════════════════
# ONNX export
# ════════════════════════════════════════════════════

def export_onnx(model, window_size, n_features, path):
    import torch

    model.eval()
    model_cpu = model.cpu()
    dummy = torch.zeros(1, window_size, n_features, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            dummy,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=18,
            dynamo=False,
        )

    print(f"\n✅ ONNX 模型已保存: {path}")


# ════════════════════════════════════════════════════
# Evidence-file writers
# ════════════════════════════════════════════════════

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv_dicts(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(path, metrics):
    rows = [
        {"true_label": "0_unintentional", "predicted_0": metrics["tn"], "predicted_1": metrics["fp"]},
        {"true_label": "1_intentional", "predicted_0": metrics["fn"], "predicted_1": metrics["tp"]},
    ]
    save_csv_dicts(path, rows)


def save_validation_metrics_csv(path, metrics):
    rows = [
        {"metric": "accuracy", "value": metrics["accuracy"]},
        {"metric": "f1_label_0_unintentional", "value": metrics["f1_0"]},
        {"metric": "f1_label_1_intentional", "value": metrics["f1_1"]},
        {"metric": "macro_f1", "value": metrics["macro_f1"]},
        {"metric": "precision_label_0_unintentional", "value": metrics["precision_0"]},
        {"metric": "recall_label_0_unintentional", "value": metrics["recall_0"]},
        {"metric": "precision_label_1_intentional", "value": metrics["precision_1"]},
        {"metric": "recall_label_1_intentional", "value": metrics["recall_1"]},
        {"metric": "tn", "value": metrics["tn"]},
        {"metric": "fp", "value": metrics["fp"]},
        {"metric": "fn", "value": metrics["fn"]},
        {"metric": "tp", "value": metrics["tp"]},
    ]
    save_csv_dicts(path, rows)


def save_stats_and_metadata(stats_path, mean, std, metrics):
    info = {
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "threshold": 0.5,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "validation_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "f1_label_0_unintentional": float(metrics["f1_0"]),
            "f1_label_1_intentional": float(metrics["f1_1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "precision_label_0_unintentional": float(metrics["precision_0"]),
            "recall_label_0_unintentional": float(metrics["recall_0"]),
            "precision_label_1_intentional": float(metrics["precision_1"]),
            "recall_label_1_intentional": float(metrics["recall_1"]),
            "confusion_matrix": {
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tp": int(metrics["tp"]),
            },
        },
        "training_note": (
            "WeightedRandomSampler was applied only to the training loader. "
            "The validation set kept the original collected distribution. "
            "Metrics are internal prototype-stage evidence, not full cross-user or cross-condition validation."
        ),
    }
    save_json(stats_path, info)
    print(f"✅ 归一化参数和验证指标已保存: {stats_path}")


def save_readme(path, args, summary, metrics):
    text = f"""Prototype-stage intention classifier evidence
=============================================

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Training script:
  train_intent_model_evidence.py

Input CSV:
  {args.csv}

Output model:
  {summary['onnx_path']}

PyTorch checkpoint:
  {summary['checkpoint_path']}

Normalisation and validation metadata:
  {summary['stats_path']}

Training log:
  {summary['log_path']}

Core features:
  {', '.join(FEATURES)}

Window configuration:
  window_size = {WINDOW_SIZE}
  step_size   = {STEP_SIZE}
  threshold   = 0.5

Dataset summary:
  raw_rows = {summary['raw_rows']}
  raw_label_0_rows = {summary['raw_label_0_rows']}
  raw_label_1_rows = {summary['raw_label_1_rows']}
  frames_after_dedup = {summary['frames_after_dedup']}
  dedup_label_0_frames = {summary['dedup_label_0_frames']}
  dedup_label_1_frames = {summary['dedup_label_1_frames']}
  skipped_inactive_label_1_frames = {summary['skipped_inactive_label_1_frames']}
  total_windows = {summary['total_windows']}
  negative_windows_label_0 = {summary['negative_windows_label_0']}
  positive_windows_label_1 = {summary['positive_windows_label_1']}
  train_windows = {summary['train_windows']}
  val_windows = {summary['val_windows']}
  train_label_0 = {summary['train_label_0']}
  train_label_1 = {summary['train_label_1']}
  val_label_0 = {summary['val_label_0']}
  val_label_1 = {summary['val_label_1']}

Best internal validation metrics:
  Accuracy = {metrics['accuracy']:.4f}
  F1(label=0, unintentional movement) = {metrics['f1_0']:.4f}
  F1(label=1, intentional control) = {metrics['f1_1']:.4f}
  Macro-F1 = {metrics['macro_f1']:.4f}
  Confusion matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}

Important limitation:
  This result is an internal prototype-stage sanity check. The dataset was
  split at segment level to reduce overlapping-window leakage. However, the
  exported dataset uses label-specific source files. Therefore, the reported
  validation metrics show separability within the collected prototype dataset,
  but they do not establish cross-user or cross-condition generalisation.

Suggested thesis wording:
  The intention classifier was trained on labelled motion-feature windows
  extracted from prototype recordings. Under internal segment-level evaluation,
  the best checkpoint achieved accuracy={metrics['accuracy']:.3f} and
  macro-F1={metrics['macro_f1']:.3f}. These results indicate that the selected
  temporal motion features were separable within the collected prototype data.
  However, because the exported data were organised into label-specific source
  files, this result should be interpreted as a prototype-stage sanity check
  rather than as conclusive evidence of source-independent or user-independent
  intention recognition.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ README 已保存: {path}")


def maybe_copy_file(src, dst):
    try:
        if src and os.path.exists(src):
            shutil.copy2(src, dst)
    except Exception as e:
        print(f"⚠️ 无法复制文件 {src} -> {dst}: {e}")


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="gesture_data.csv")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--evidence_dir", default="intent_model_evidence")
    parser.add_argument("--copy_csv", action="store_true", help="Copy input CSV into evidence_dir (may be large).")
    parser.add_argument("--out", default=None, help="Optional ONNX output path. Default: evidence_dir/intent_model.onnx")
    args = parser.parse_args()

    ensure_dir(args.evidence_dir)
    evidence_dir = Path(args.evidence_dir)

    log_path = evidence_dir / "intent_training_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        set_seed(RANDOM_SEED)

        if not os.path.exists(args.csv):
            print(f"❌ 找不到 CSV 文件: {args.csv}")
            print("   建议使用绝对路径，例如：")
            print(r"   python train_intent_model_evidence.py --csv E:\mijia-api\gesture_data.csv")
            return

        try:
            import torch
        except ImportError:
            print("❌ 请先安装 PyTorch：pip install torch")
            return

        print("=" * 68)
        print("  手势意图模型训练（证据包输出版）")
        print("=" * 68)
        print(f"Evidence directory: {evidence_dir.resolve()}")
        print(f"Random seed: {RANDOM_SEED}")
        print(f"Window size: {WINDOW_SIZE}")
        print(f"Step size: {STEP_SIZE}")
        print(f"Features: {FEATURES}")

        # Copy script for evidence if possible
        try:
            script_path = Path(__file__).resolve()
            maybe_copy_file(script_path, evidence_dir / "train_intent_model_evidence.py")
        except Exception:
            pass

        if args.copy_csv:
            maybe_copy_file(args.csv, evidence_dir / Path(args.csv).name)

        df_raw, df_clean, skipped = load_data(args.csv)
        X, y, groups, group_meta = make_windows(df_clean)

        if len(X) < 30:
            print(f"❌ 样本太少（{len(X)}），至少需要 30 个窗口")
            return

        X_train, y_train, X_val, y_val, split_meta = split_data(
            X,
            y,
            groups,
            group_meta,
            val_ratio=args.val_ratio,
        )

        X_train, X_val, mean, std = normalize(X_train, X_val)

        model = build_model(len(FEATURES))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n📐 模型参数量: {n_params:,}")

        model, best_metrics, history = train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        onnx_path = Path(args.out) if args.out else evidence_dir / "intent_model.onnx"
        checkpoint_path = evidence_dir / "intent_model_best.pth"
        stats_path = evidence_dir / "intent_model_stats.json"
        dataset_summary_path = evidence_dir / "dataset_summary.json"
        split_path = evidence_dir / "group_split_summary.csv"
        history_path = evidence_dir / "training_history.csv"
        confusion_path = evidence_dir / "confusion_matrix.csv"
        metrics_path = evidence_dir / "validation_metrics.csv"
        readme_path = evidence_dir / "README_intention_model.txt"

        # Save checkpoint
        torch.save({
            "model_state_dict": model.cpu().state_dict(),
            "features": FEATURES,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "threshold": 0.5,
            "validation_metrics": best_metrics,
            "random_seed": RANDOM_SEED,
        }, checkpoint_path)
        print(f"✅ PyTorch checkpoint 已保存: {checkpoint_path}")

        # Save metadata and evidence files
        save_stats_and_metadata(stats_path, mean, std, best_metrics)
        save_csv_dicts(history_path, history)
        print(f"✅ 训练历史已保存: {history_path}")
        save_confusion_matrix_csv(confusion_path, best_metrics)
        print(f"✅ 混淆矩阵已保存: {confusion_path}")
        save_validation_metrics_csv(metrics_path, best_metrics)
        print(f"✅ 验证指标已保存: {metrics_path}")
        save_csv_dicts(split_path, split_meta)
        print(f"✅ group 划分摘要已保存: {split_path}")

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "csv_path": str(args.csv),
            "evidence_dir": str(evidence_dir.resolve()),
            "onnx_path": str(onnx_path),
            "checkpoint_path": str(checkpoint_path),
            "stats_path": str(stats_path),
            "log_path": str(log_path),
            "random_seed": RANDOM_SEED,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "features": FEATURES,
            "raw_rows": int(len(df_raw)),
            "raw_label_0_rows": int((df_raw["label"] == 0).sum()),
            "raw_label_1_rows": int((df_raw["label"] == 1).sum()),
            "raw_sources": [str(x) for x in sorted(df_raw["source"].unique().tolist())],
            "raw_segment_count": int(df_raw[["source", "segment"]].drop_duplicates().shape[0]),
            "frames_after_dedup": int(len(df_clean)),
            "dedup_label_0_frames": int((df_clean["label"] == 0).sum()),
            "dedup_label_1_frames": int((df_clean["label"] == 1).sum()),
            "skipped_inactive_label_1_frames": int(skipped),
            "total_windows": int(len(X)),
            "negative_windows_label_0": int((1 - y).sum()),
            "positive_windows_label_1": int(y.sum()),
            "train_windows": int(len(X_train)),
            "val_windows": int(len(X_val)),
            "train_label_0": int((1 - y_train).sum()),
            "train_label_1": int(y_train.sum()),
            "val_label_0": int((1 - y_val).sum()),
            "val_label_1": int(y_val.sum()),
            "n_params": int(n_params),
            "validation_metrics": {
                "accuracy": float(best_metrics["accuracy"]),
                "f1_label_0_unintentional": float(best_metrics["f1_0"]),
                "f1_label_1_intentional": float(best_metrics["f1_1"]),
                "macro_f1": float(best_metrics["macro_f1"]),
                "tn": int(best_metrics["tn"]),
                "fp": int(best_metrics["fp"]),
                "fn": int(best_metrics["fn"]),
                "tp": int(best_metrics["tp"]),
            },
            "evidence_boundary": (
                "Internal prototype-stage sanity check. Segment-level split reduces overlapping-window leakage, "
                "but exported label-specific source files do not establish cross-user or cross-condition generalisation."
            ),
        }
        save_json(dataset_summary_path, summary)
        print(f"✅ 数据集摘要已保存: {dataset_summary_path}")

        save_readme(readme_path, args, summary, best_metrics)

        export_onnx(model, WINDOW_SIZE, len(FEATURES), onnx_path)

        print("\n" + "=" * 68)
        print("  训练完成：证据包已生成")
        print(f"  Evidence dir   : {evidence_dir.resolve()}")
        print(f"  ONNX model     : {onnx_path}")
        print(f"  Checkpoint     : {checkpoint_path}")
        print(f"  Stats JSON     : {stats_path}")
        print(f"  Training log   : {log_path}")
        print(f"  Accuracy       : {best_metrics['accuracy']:.4f}")
        print(f"  F1 label=0     : {best_metrics['f1_0']:.4f}  # 无意动作")
        print(f"  F1 label=1     : {best_metrics['f1_1']:.4f}  # 有意调节")
        print(f"  Macro-F1       : {best_metrics['macro_f1']:.4f}")
        print(
            f"  Confusion      : "
            f"TN={best_metrics['tn']} FP={best_metrics['fp']} "
            f"FN={best_metrics['fn']} TP={best_metrics['tp']}"
        )
        print("=" * 68)

    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()
