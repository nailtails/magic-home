from collections import deque

import numpy as np
import torch

from config import Config
from lstm_model import LSTMClassifier
from runtime_utils import FEATURE_COLS, build_feature_vector_from_live_frame, load_device_positions


# =========================================================
# 1. 基础工具
# =========================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def state_to_name(state: int) -> str:
    mapping = {
        0: "OFF",
        1: "ON_DEVICE1",
        2: "ON_DEVICE2",
    }
    return mapping.get(state, f"UNKNOWN_{state}")


def _torch_load_compat(model_path: str, device):
    """
    兼容不同 PyTorch 版本的 torch.load。
    """
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _looks_like_state_dict(obj) -> bool:
    """
    判断一个对象是否像 PyTorch state_dict。
    """
    if not isinstance(obj, dict) or len(obj) == 0:
        return False

    first_key = next(iter(obj.keys()))
    first_val = obj[first_key]
    return isinstance(first_key, str) and torch.is_tensor(first_val)


def _extract_model_kwargs(ckpt, input_size: int, cfg: Config) -> dict:
    """
    优先使用 checkpoint 中保存的 model_kwargs；
    如果没有，就退回到当前 Config。
    """
    default_kwargs = {
        "input_size": input_size,
        "hidden_size": cfg.lstm.hidden_size,
        "num_layers": cfg.lstm.num_layers,
        "num_classes": cfg.task.num_classes,
        "bidirectional": cfg.lstm.bidirectional,
        "dropout": cfg.lstm.dropout,
        "fc_hidden_size": cfg.lstm.fc_hidden_size,
    }

    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_kwargs"), dict):
        model_kwargs = {**default_kwargs, **ckpt["model_kwargs"]}
    else:
        model_kwargs = default_kwargs

    if int(model_kwargs["input_size"]) != int(input_size):
        raise ValueError(
            f"模型 input_size 与当前实时特征维度不一致："
            f"{model_kwargs['input_size']} != {input_size}"
        )

    return model_kwargs


def _extract_state_dict(ckpt):
    """
    兼容多种 checkpoint 格式。
    """
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "net_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        if _looks_like_state_dict(ckpt):
            return ckpt

    if _looks_like_state_dict(ckpt):
        return ckpt

    raise ValueError(
        "无法从 checkpoint 中解析 state_dict。"
        "请检查 .pth 文件格式，或把 checkpoint 的 keys 打印出来确认。"
    )


def _check_feature_cols_if_possible(ckpt):
    """
    如果 checkpoint 中保存了 feature_cols，就做严格一致性校验。
    """
    if not isinstance(ckpt, dict):
        return

    ckpt_feature_cols = ckpt.get("feature_cols", None)
    if ckpt_feature_cols is None:
        return

    ckpt_feature_cols = list(ckpt_feature_cols)
    current_feature_cols = list(FEATURE_COLS)

    if ckpt_feature_cols != current_feature_cols:
        raise ValueError(
            "当前实时 FEATURE_COLS 与训练该 finetune 模型时使用的 feature_cols 不一致！\n"
            f"checkpoint feature_cols 长度 = {len(ckpt_feature_cols)}\n"
            f"runtime    FEATURE_COLS 长度 = {len(current_feature_cols)}"
        )


def load_model(model_path: str, input_size: int, device, cfg: Config):
    ckpt = _torch_load_compat(model_path, device)

    _check_feature_cols_if_possible(ckpt)

    model_kwargs = _extract_model_kwargs(
        ckpt=ckpt,
        input_size=input_size,
        cfg=cfg,
    )

    model = LSTMClassifier(
        input_size=model_kwargs["input_size"],
        hidden_size=model_kwargs["hidden_size"],
        num_layers=model_kwargs["num_layers"],
        num_classes=model_kwargs["num_classes"],
        bidirectional=model_kwargs["bidirectional"],
        dropout=model_kwargs["dropout"],
        fc_hidden_size=model_kwargs["fc_hidden_size"],
    ).to(device)

    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, model_kwargs


# =========================================================
# 2. 实时平滑器：EMA
# =========================================================

class EMASmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = x.copy()
        else:
            self.prev = self.alpha * x + (1.0 - self.alpha) * self.prev
        return self.prev.copy()


# =========================================================
# 3. 模型输出稳定器
# =========================================================

class PredictionStabilizer:
    def __init__(self, prob_window=5, conf_thresh=0.60, switch_patience=3):
        self.prob_buffer = deque(maxlen=prob_window)
        self.conf_thresh = conf_thresh
        self.switch_patience = switch_patience

        self.current_state = 0
        self.pending_state = None
        self.pending_count = 0

    def update(self, prob: np.ndarray):
        self.prob_buffer.append(prob)
        avg_prob = np.mean(self.prob_buffer, axis=0)

        pred = int(np.argmax(avg_prob))
        conf = float(np.max(avg_prob))

        if conf < self.conf_thresh:
            pred = 0

        if pred == self.current_state:
            self.pending_state = None
            self.pending_count = 0
        else:
            if self.pending_state == pred:
                self.pending_count += 1
            else:
                self.pending_state = pred
                self.pending_count = 1

            if self.pending_count >= self.switch_patience:
                self.current_state = pred
                self.pending_state = None
                self.pending_count = 0

        return self.current_state, avg_prob


# =========================================================
# 4. 字幕显示稳定器
# =========================================================

class DisplayStateController:
    def __init__(self, hold_count=3):
        self.hold_count = hold_count
        self.current_display_state = 0
        self.candidate_state = None
        self.candidate_count = 0

    def update(self, new_state: int) -> int:
        if new_state == self.current_display_state:
            self.candidate_state = None
            self.candidate_count = 0
            return self.current_display_state

        if self.candidate_state == new_state:
            self.candidate_count += 1
        else:
            self.candidate_state = new_state
            self.candidate_count = 1

        if self.candidate_count >= self.hold_count:
            self.current_display_state = new_state
            self.candidate_state = None
            self.candidate_count = 0

        return self.current_display_state


# =========================================================
# 5. 实时预测器
# =========================================================

class RealtimeAttentionPredictor:
    def __init__(
        self,
        model_path: str,
        device_json_path: str,
        ema_alpha: float = 0.5,
        use_ema: bool = True,
        prob_window: int = 5,
        conf_thresh: float = 0.60,
        switch_patience: int = 3,
        display_hold_count: int = 3,
    ):
        self.config = Config()
        self.device = get_device()

        self.window_size = self.config.data.window_size
        self.stride = self.config.data.stride

        self.model, self.model_kwargs = load_model(
            model_path=model_path,
            input_size=len(FEATURE_COLS),
            device=self.device,
            cfg=self.config,
        )

        self.device1_xyz, self.device2_xyz = load_device_positions(device_json_path)

        self.use_ema = use_ema
        self.smoother = EMASmoother(alpha=ema_alpha)

        self.stabilizer = PredictionStabilizer(
            prob_window=prob_window,
            conf_thresh=conf_thresh,
            switch_patience=switch_patience,
        )

        self.display_controller = DisplayStateController(
            hold_count=display_hold_count
        )

        self.frame_buffer = deque(maxlen=self.window_size)
        self.frame_count = 0

    def update(self, joints_xyz: dict):
        feat = build_feature_vector_from_live_frame(
            joints_xyz=joints_xyz,
            device1_xyz=self.device1_xyz,
            device2_xyz=self.device2_xyz,
        )

        if self.use_ema:
            feat = self.smoother.update(feat)

        self.frame_buffer.append(feat)
        self.frame_count += 1

        if len(self.frame_buffer) < self.window_size:
            return None

        if self.frame_count % self.stride != 0:
            return None

        x = np.array(self.frame_buffer, dtype=np.float32)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

        raw_pred = int(np.argmax(prob))
        stable_pred, avg_prob = self.stabilizer.update(prob)
        display_pred = self.display_controller.update(stable_pred)

        return {
            "frame_count": self.frame_count,
            "raw_pred": raw_pred,
            "raw_name": state_to_name(raw_pred),
            "stable_pred": stable_pred,
            "stable_name": state_to_name(stable_pred),
            "display_pred": display_pred,
            "display_name": state_to_name(display_pred),
            "raw_prob": prob,
            "avg_prob": avg_prob,
        }