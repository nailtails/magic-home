import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import gesture_controller_simulated as gc
from predictor_core import RealtimeAttentionPredictor
from runtime_utils import KinectRuntime

# 不再使用固定 2 秒锁死；是否允许切换由 CONTROL_END_SEC 判断
CONTROL_SESSION_SEC = 0.0
# 不再使用控制结束保护；realtime intent 可以每帧判断目标
CONTROL_END_SEC = 0.0
# 不再使用真实控制动作后的设备切换保护；设备可以根据 realtime intent 立即切换
ACTION_SWITCH_PROTECT_SEC = 0.0
# Kinect intent 检测间隔：0.0 表示控制结束后每帧实时检测下一目标
KINECT_RECHECK_SEC = 0.0
# 控制会话中超过该时间没有真实控制动作，则释放 MediaPipe，回到 Kinect-only 模式
CONTROL_IDLE_EXIT_SEC = 5.0

# =========================================================
# Realtime attention 参数：严格对齐单独 realtime(2).py
# =========================================================
# 说明：
# 0 = OFF
# 1 = ON_DEVICE1
# 2 = ON_DEVICE2
#
# 这组参数与当前单独 realtime 测试文件保持一致：
# - 优先使用 GCN-Att-LSTM checkpoint
# - 保留 Device2 的轻量正向 bias
# - 保留侧身/肩点重合时的手臂关键点自适应增权
DEFAULT_CLASS_CONF_THRESH = {
    0: 0.55,
    1: 0.60,
    2: 0.80
}

DEFAULT_CLASS_SWITCH_PATIENCE = {
    0: 2,
    1: 2,
    2: 2,
}

DEFAULT_HOLD_BY_STATE = {
    0: 1,
    1: 2,
    2: 1,
}

DEFAULT_LOGIT_BIAS = [
    0.0,    # OFF
    0.0,  # ON_DEVICE1
    0.0# ON_DEVICE2
]

# 侧身导致肩点重合时，对手臂关键点做自适应增权
DEFAULT_SHOULDER_OVERLAP_RANGE_M = (0.02, 0.04)
DEFAULT_ARM_BOOST_MIN = 1.12
DEFAULT_ARM_BOOST_MAX = 1.45

# GestureRecognizer 隔帧运行，减轻负载
GR_EVERY_N_FRAMES = 3

GESTURE_TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")

# Mediapipe 调优
POSE_SCALE = 0.5              # Pose 用半分辨率，减轻负载
HANDS_DET_CONF = 0.40        # 提高 tracking 稳定性，减少远距离误检/漂移
HANDS_TRK_CONF = 0.55
GESTURE_MIN_VISIBLE = 4
GESTURE_STABLE_FRAMES = 1
GESTURE_LOST_GRACE = 10
GESTURE_HOLD_SEC_OVERRIDE = 0.14
SEQ_FRAMES_OVERRIDE = 3
SEQ_WINDOW_OVERRIDE = 2.4
UI_REFRESH_EVERY = 2         # 面板文本隔帧刷新

# ROI 放大识别
ROI_TARGET_SIZE = 640
ROI_MARGIN_X = 0.55
ROI_MARGIN_TOP = 0.55
ROI_MARGIN_BOTTOM = 0.50
ROI_FALLBACK_KEEP = 45
SHOW_ROI_BOX = True

# ROI 失败兜底：如果 ROI 内没有识别到手，定期用全图 Hands 检测一次。
# 主要解决手放身前时 Pose-based ROI 没包住手的问题。
FULLFRAME_HAND_FALLBACK_EVERY = 5

# UI 显示开关：不显示 MediaPipe Pose / Hands 骨架，只显示手势文字
DRAW_MEDIAPIPE_POSE_SKELETON = False
DRAW_HAND_CONNECTIONS = False  # 隐藏手部骨骼连接线
DRAW_HAND_POINTS = True        # 显示手掌和手指关键点
DRAW_GESTURE_TEXT = True       # 显示 Open/Fist 文字

# 手部跟踪容错：
# 主循环中如果某一帧 Hands 没检测到手，不立刻判定丢失；
# 在 HAND_LOST_GRACE_OVERRIDE 帧内继续认为该手处于 ready，减少连续控制断裂。
HAND_READY_MIN_VISIBLE = 4
HAND_LOST_GRACE_OVERRIDE = 10

# GestureRecognizer 只做辅助判断，几何规则作为 open/fist 主干。
USE_GESTURE_RECOGNIZER_AS_FALLBACK = True

# 如果当前 Open/Fist 的 on/off 顺序和实际预期相反，打开这个开关。
# True 表示 GestureTracker 输出 on 时按 off 执行，输出 off 时按 on 执行。
INVERT_OPEN_FIST_ACTION = True

# 侧面手掌/拳头兜底：当原几何规则和 GestureRecognizer 不稳定时，
# 额外用指尖展开程度判断 Open/Fist。
USE_SIDE_OPEN_FIST_FALLBACK = True

# Open/Fist 开关逻辑：
# True 表示使用方向式 OPEN/FIST 转换开关：
# OPEN -> FIST = 关
# FIST -> OPEN = 开
# 中间短暂 UNKNOWN 不会打断动作。
USE_TRANSITION_TOGGLE_SWITCH = True
USE_DIRECTIONAL_OPEN_FIST_SWITCH = True
GESTURE_TOGGLE_COOLDOWN_SEC = 1.0
# 必须在该时间窗口内完成 OPEN <-> FIST 转换，才触发方向式开关。
# 超过 3 秒则只更新起始手势，不执行开关；中间 UNKNOWN 不会立即打断。
GESTURE_TRANSITION_WINDOW_SEC = 3.0


def resolve_device_json_path():
    """
    优先找新名字 device_positions.json；
    如果没有，再兼容旧名字 device_coords.json。
    """
    candidates = [
        "device_positions.json",
        "device_coords.json",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return "device_positions.json"


def resolve_model_path():
    """
    与单独 realtime(2).py 对齐：
    优先使用 GCN-Att-LSTM checkpoint，找不到再回退到 LSTM checkpoint。
    """
    candidates = [
        "best_gcn_att_lstm_model.pth",
        "/Users/zjy/Desktop/威力加强版_副本/outputs/best_gcn_att_lstm_model.pth",
        "lstm_best_model.pth",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def load_gesture_recognizer():
    if not os.path.exists(GESTURE_TASK_PATH):
        print(f"⚠️ 未找到 {GESTURE_TASK_PATH}，open/fist 仅使用几何规则")
        return None

    with open(GESTURE_TASK_PATH, "rb") as f:
        task_bytes = f.read()

    opts = mp_vision.GestureRecognizerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=task_bytes),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rec = mp_vision.GestureRecognizer.create_from_options(opts)
    print(f"✅ GestureRecognizer 已加载(buffer): {GESTURE_TASK_PATH}")
    return rec


def put(img, text, x, y, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)




def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_pose_roi_from_small(frame_shape, pose_landmarks, pose_scale, last_roi=None):
    if pose_landmarks is None:
        return last_roi

    h, w = frame_shape[:2]
    lm = pose_landmarks.landmark
    idx = mp.solutions.pose.PoseLandmark
    needed = [
        idx.LEFT_SHOULDER, idx.RIGHT_SHOULDER,
        idx.LEFT_ELBOW, idx.RIGHT_ELBOW,
        idx.LEFT_WRIST, idx.RIGHT_WRIST,
    ]

    pts = []
    for i in needed:
        p = lm[i]
        if getattr(p, 'visibility', 1.0) < 0.25:
            continue
        x = (p.x * w * pose_scale) / pose_scale
        y = (p.y * h * pose_scale) / pose_scale
        if 0 <= x < w and 0 <= y < h:
            pts.append([x, y])

    if len(pts) < 3:
        return last_roi

    pts = np.array(pts, dtype=np.float32)
    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return last_roi

    x1 -= bw * ROI_MARGIN_X
    x2 += bw * ROI_MARGIN_X
    y1 -= bh * ROI_MARGIN_TOP
    y2 += bh * ROI_MARGIN_BOTTOM

    x1 = int(clamp(x1, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    x2 = int(clamp(x2, 1, w))
    y2 = int(clamp(y2, 1, h))

    if x2 - x1 < 60 or y2 - y1 < 60:
        return last_roi
    return (x1, y1, x2, y2)


def crop_and_resize_roi(frame, roi):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    ch, cw = crop.shape[:2]
    scale = ROI_TARGET_SIZE / max(ch, cw)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (cw, ch, new_w, new_h)


def roi_point_to_full(nx, ny, roi, resize_info):
    x1, y1, x2, y2 = roi
    cw, ch, new_w, new_h = resize_info
    px_roi = nx * new_w
    py_roi = ny * new_h
    px_full = x1 + (px_roi / new_w) * cw
    py_full = y1 + (py_roi / new_h) * ch
    return float(px_full), float(py_full)



def fullframe_resize_info(frame):
    """
    全图 Hands 兜底时使用。
    由于 Hands 直接在完整 frame 上运行，remap 时等价于原图宽高。
    """
    h, w = frame.shape[:2]
    return (w, h, w, h)


def remap_hand_landmarks_to_full(hlm, roi, resize_info, frame_shape):
    h, w = frame_shape[:2]
    out = []
    for lmk in hlm.landmark:
        fx, fy = roi_point_to_full(lmk.x, lmk.y, roi, resize_info)
        out.append((fx / w, fy / h))
    return out



def classify_open_fist_side_fallback(hlm):
    """
    侧面/斜侧面手势兜底分类器。

    原理：
    - 使用 21 个 MediaPipe hand landmarks。
    - 以 wrist 到 middle_mcp 的距离估计手掌尺度。
    - 比较各指尖 tip 到 palm_center 的距离与对应 pip 到 palm_center 的距离。
    - 伸展手指数量 >= 4 判定为 Open。
    - 伸展手指数量 <= 1 且指尖整体靠近掌心，判定为 Fist。

    这个规则不依赖手掌正对摄像头，比单纯 y 方向规则更适合侧面手势。
    """
    try:
        pts = np.array([[p.x, p.y] for p in hlm.landmark], dtype=np.float32)
        if pts.shape[0] < 21:
            return False, False, 0

        wrist = pts[0]
        palm_ids = [0, 5, 9, 13, 17]
        palm_center = pts[palm_ids].mean(axis=0)

        palm_size = float(np.linalg.norm(pts[9] - wrist))
        if palm_size < 1e-6:
            return False, False, 0

        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        mcps = [2, 5, 9, 13, 17]

        extended = 0
        close_to_palm = 0

        for tip, pip, mcp in zip(tips, pips, mcps):
            d_tip = float(np.linalg.norm(pts[tip] - palm_center))
            d_pip = float(np.linalg.norm(pts[pip] - palm_center))
            d_mcp = float(np.linalg.norm(pts[mcp] - palm_center))

            # tip 明显比 pip 更远，认为该手指伸展。
            if d_tip > d_pip + 0.10 * palm_size and d_tip > d_mcp + 0.08 * palm_size:
                extended += 1

            # tip 靠近掌心，认为该手指收拢。
            if d_tip < d_pip + 0.08 * palm_size:
                close_to_palm += 1

        side_open = extended >= 4
        side_fist = extended <= 1 and close_to_palm >= 3

        return side_open, side_fist, extended

    except Exception:
        return False, False, 0


def get_target_pred_from_result(result, prefer_key="stable_pred"):
    """
    用于控制目标切换的预测结果读取。
    默认使用 stable_pred，比 display_pred 更灵敏，但比 raw_pred 稳定。
    如果某些旧 predictor 没有 stable_pred，则自动回退到 display_pred。
    """
    if result is None:
        return None
    if prefer_key in result:
        return result.get(prefer_key)
    return result.get("display_pred")

def draw_intent_status_box(frame, result, active_target, has_body, mediapipe_on, control_until, intent_recheck_sec=KINECT_RECHECK_SEC):
    """
    中央状态框：
    - 不和左右 gc.draw_panel 的面板重叠
    - 空闲态显示 intent 结果
    - 控制态显示会话剩余时间
    """
    H, W = frame.shape[:2]

    box_w = 450
    box_h = 150
    x1 = (W - box_w) // 2
    y1 = 12
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    mode_text = "MODE: INTENT ONLY"
    mode_color = (160, 160, 160)
    if mediapipe_on and active_target == 1:
        mode_text = "MODE: CONTROL DEVICE1"
        mode_color = (255, 220, 80)
    elif mediapipe_on and active_target == 2:
        mode_text = "MODE: CONTROL DEVICE2"
        mode_color = (120, 210, 255)

    body_text = "BODY: YES" if has_body else "BODY: NO"
    body_color = (0, 220, 120) if has_body else (120, 120, 120)

    put(frame, mode_text, x1 + 12, y1 + 24, 0.62, mode_color, 2)
    put(frame, body_text, x2 - 110, y1 + 24, 0.48, body_color, 1)

    target_name = {None: "SLEEP", 1: "DEVICE1", 2: "DEVICE2"}.get(active_target, "UNKNOWN")
    put(frame, f"active target: {target_name}", x1 + 12, y1 + 52, 0.46, (220, 220, 220), 1)

    if mediapipe_on:
        left = max(0.0, control_until - time.time())
        put(frame, f"switch delay: {left:.1f}s", x1 + 12, y1 + 80, 0.48, (0, 220, 120), 1)
        if intent_recheck_sec <= 0:
            put(frame, "Kinect intent realtime switching", x1 + 12, y1 + 110, 0.42, (160, 160, 160), 1)
        else:
            put(frame, f"Kinect intent recheck every {intent_recheck_sec:.1f}s during control", x1 + 12, y1 + 110, 0.42, (160, 160, 160), 1)
        put(frame, "No action-switch protection; stable_pred switches immediately", x1 + 12, y1 + 136, 0.37, (140, 140, 140), 1)
        return

    if result is None:
        put(frame, "intent: waiting for enough frames...", x1 + 12, y1 + 84, 0.48, (185, 185, 185), 1)
        put(frame, "Mediapipe is OFF in idle mode", x1 + 12, y1 + 112, 0.42, (150, 150, 150), 1)
        return

    # 新版 RealtimeAttentionPredictor 返回字段：
    # raw_name_model/raw_prob_model = 纯模型输出
    # raw_name/raw_prob             = 加 logit_bias 后的即时输出
    # stable_name/avg_prob          = 概率滑窗 + 阈值 + patience 后的稳定输出
    # display_name/display_pred     = 最终显示与控制层输出
    model_name = result.get("raw_name_model", "N/A")
    biased_name = result.get("raw_name", "N/A")
    stable_name = result.get("stable_name", "N/A")
    display_name = result.get("display_name", "N/A")
    raw_prob = result.get("raw_prob", None)
    avg_prob = result.get("avg_prob", None)

    put(frame, f"model   : {model_name}", x1 + 12, y1 + 78, 0.46, (235, 235, 235), 1)
    put(frame, f"biased  : {biased_name}", x1 + 12, y1 + 100, 0.46, (220, 220, 220), 1)
    put(frame, f"stable  : {stable_name}", x1 + 12, y1 + 122, 0.46, (210, 210, 210), 1)
    put(frame, f"display : {display_name}", x1 + 12, y1 + 144, 0.46, (0, 220, 120), 1)

    if raw_prob is not None and len(raw_prob) >= 3:
        p0, p1, p2 = raw_prob[:3]
        prob_text = f"B=[{p0:.3f}, {p1:.3f}, {p2:.3f}]"
        put(frame, prob_text, x1 + 225, y1 + 78, 0.40, (160, 160, 160), 1)

    if avg_prob is not None and len(avg_prob) >= 3:
        p0, p1, p2 = avg_prob[:3]
        avg_text = f"S=[{p0:.3f}, {p1:.3f}, {p2:.3f}]"
        put(frame, avg_text, x1 + 225, y1 + 100, 0.40, (160, 160, 160), 1)


def apply_control_for_any_hand(
    ctrl: gc.ArmController,
    side_label: str,
    active_target: int,
    state: gc.VirtualState,
    worker: gc.MijiaWorker,
    hand_ready: dict,
    hand_visible: dict,
    dt: float,
    session_base_b,
    session_base_v,
    session_base_t,
    prev_x_axis_trigger: dict,
    now: float,
    last_log_b: float,
    last_log_v: float,
):
    """
    返回：
      session_base_b, session_base_v, session_base_t, last_log_b, last_log_v, action_used
    action_used=True 表示发生了真实控制动作，可用于续期控制会话
    """
    action_used = False

    if not ctrl.active or not hand_ready[side_label]:
        return session_base_b, session_base_v, session_base_t, last_log_b, last_log_v, action_used

    dv = ctrl.calc_delta_value()
    hv = hand_visible[side_label]

    # 设备处于 OFF 状态时，不允许继续调节参数；只允许用 open/fist 去开/关设备本身
    if active_target == 1 and not getattr(state, "lamp_on", False):
        prev_x_axis_trigger[side_label] = ctrl.axis
        return session_base_b, session_base_v, session_base_t, last_log_b, last_log_v, action_used
    if active_target == 2 and not getattr(state, "speaker_playing", False):
        prev_x_axis_trigger[side_label] = ctrl.axis
        return session_base_b, session_base_v, session_base_t, last_log_b, last_log_v, action_used

    if active_target == 1:
        # Device1 -> lamp: y=brightness, x=temp
        if ctrl.axis == "y" and dv != 0.0 and hv >= gc.HAND_MIN_VISIBLE_Y:
            raw_b = state.brightness + dv
            new_b = gc.ArmController.rate_limit(state.brightness, raw_b, gc.MAX_RATE_B, dt)
            new_b = gc.clamp(new_b, gc.B_MIN, gc.B_MAX)
            new_b = ctrl.dead_zone(state.brightness, new_b, gc.DEAD_ZONE_B)

            if session_base_b is not None:
                if new_b > session_base_b + gc.SESSION_CAP_B and dv > 0:
                    new_b = session_base_b + gc.SESSION_CAP_B
                if new_b < session_base_b - gc.SESSION_CAP_B and dv < 0:
                    new_b = session_base_b - gc.SESSION_CAP_B

            if abs(new_b - state.brightness) > 1e-6:
                state.brightness = new_b
                worker.send("brightness", state.brightness)
                action_used = True
                if now - last_log_b > 0.4:
                    state.push_log(f"Brightness = {int(state.brightness)}%")
                    last_log_b = now

        elif ctrl.axis == "x" and dv != 0.0 and hv >= gc.HAND_MIN_VISIBLE_X:
            raw_t = state.temp - dv * 40.0
            new_t = gc.ArmController.rate_limit(state.temp, raw_t, gc.MAX_RATE_T, dt)
            new_t = gc.clamp(new_t, gc.T_MIN, gc.T_MAX)
            new_t = ctrl.dead_zone(state.temp, new_t, gc.DEAD_ZONE_T)

            if session_base_t is not None:
                if new_t > session_base_t + gc.SESSION_CAP_T and dv > 0:
                    new_t = session_base_t + gc.SESSION_CAP_T
                if new_t < session_base_t - gc.SESSION_CAP_T and dv < 0:
                    new_t = session_base_t - gc.SESSION_CAP_T

            if abs(new_t - state.temp) > 1e-6:
                state.temp = new_t
                worker.send("temp", state.temp)
                action_used = True
                if now - last_log_b > 0.4:
                    state.push_log(f"Temp = {int(state.temp)}K")
                    last_log_b = now

    elif active_target == 2:
        # Device2 -> speaker: y=volume, x=track
        if ctrl.axis == "y" and dv != 0.0 and hv >= gc.HAND_MIN_VISIBLE_Y:
            raw_v = state.volume + dv
            new_v = gc.ArmController.rate_limit(state.volume, raw_v, gc.MAX_RATE_V, dt)
            new_v = gc.clamp(new_v, 0, 100)
            new_v = ctrl.dead_zone(state.volume, new_v, gc.DEAD_ZONE_V)

            if session_base_v is not None:
                if new_v > session_base_v + gc.SESSION_CAP_V and dv > 0:
                    new_v = session_base_v + gc.SESSION_CAP_V
                if new_v < session_base_v - gc.SESSION_CAP_V and dv < 0:
                    new_v = session_base_v - gc.SESSION_CAP_V

            if abs(new_v - state.volume) > 1e-6:
                state.volume = new_v
                worker.send("volume", state.volume)
                action_used = True
                if now - last_log_v > 0.4:
                    state.push_log(f"Volume = {round(state.volume, 1)}%")
                    last_log_v = now

        elif ctrl.axis == "x" and dv != 0.0 and prev_x_axis_trigger[side_label] != "x":
            if ctrl.delta > 0:
                state.track += 1
                state.push_log(f"Next Track -> #{state.track}")
                worker.send("next_track", 1)
            else:
                state.track = max(1, state.track - 1)
                state.push_log(f"Prev Track -> #{state.track}")
                worker.send("prev_track", 1)

            action_used = True

    prev_x_axis_trigger[side_label] = ctrl.axis
    return session_base_b, session_base_v, session_base_t, last_log_b, last_log_v, action_used


def choose_primary_control_hand(lc, rc, hand_ready, hand_visible):
    """
    合并左右手功能：任意一只当前更稳定/更清晰的手都可以控制当前 active_target。
    返回 (ctrl, side_label, dt) 或 (None, None, None)
    """
    candidates = []
    for ctrl, side in ((lc, "Left"), (rc, "Right")):
        if ctrl.active and hand_ready.get(side, False):
            hv = hand_visible.get(side, 0)
            move = abs(getattr(ctrl, 'delta', 0.0))
            candidates.append((hv, move, ctrl.last_t if hasattr(ctrl, 'last_t') else 0.0, ctrl, side))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    _, _, _, ctrl, side = candidates[0]
    return ctrl, side, ctrl.get_dt()


def toggle_active_device(active_target, state, worker):
    """
    Open/Fist 状态切换式开关：
    - OPEN -> FIST 或 FIST -> OPEN 都触发一次 toggle
    - 如果当前设备是开，则关闭
    - 如果当前设备是关，则打开

    返回 True 表示确实执行了控制动作。
    """
    if active_target == 1:
        # 保持原代码映射：active_target == 1 使用 lamp_on / lamp_off
        if getattr(state, "lamp_on", False):
            state.lamp_on = False
            state.push_log("Lamp OFF")
            worker.send("lamp_on", False)
        else:
            state.lamp_on = True
            state.push_log("Lamp ON")
            worker.send("lamp_on", True)
        return True

    if active_target == 2:
        # 保持原代码映射：active_target == 2 使用 speaker play/stop
        if getattr(state, "speaker_playing", False):
            state.speaker_playing = False
            state.push_log("Speaker STOP")
            worker.send("stop", 1)
        else:
            state.speaker_playing = True
            state.push_log("Speaker PLAY")
            worker.send("play", 1)
        return True

    return False


def set_active_device_switch(active_target, target_on, state, worker):
    """
    方向式 OPEN/FIST 开关：
    - FIST -> OPEN = 开
    - OPEN -> FIST = 关

    target_on=True  表示执行开
    target_on=False 表示执行关

    返回 True 表示确实执行了控制动作。
    """
    if active_target == 1:
        if target_on:
            state.lamp_on = True
            state.push_log("Lamp ON")
            worker.send("lamp_on", True)
        else:
            state.lamp_on = False
            state.push_log("Lamp OFF")
            worker.send("lamp_on", False)
        return True

    if active_target == 2:
        if target_on:
            state.speaker_playing = True
            state.push_log("Speaker PLAY")
            worker.send("play", 1)
        else:
            state.speaker_playing = False
            state.push_log("Speaker STOP")
            worker.send("stop", 1)
        return True

    return False


def run(worker: gc.MijiaWorker, predictor: RealtimeAttentionPredictor, kinect_runtime: KinectRuntime):
    gc._load_intent_model()

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Idle 时不启 Mediapipe，进入控制会话后再启
    pose = None
    hands = None
    gesture_recognizer = None

    lc = gc.ArmController("left")
    rc = gc.ArmController("right")
    gest = gc.GestureTracker()
    # 针对侧边 / 转换抖动，提升序列确认门槛
    gest._fist_buf = {"Left": deque(maxlen=SEQ_FRAMES_OVERRIDE), "Right": deque(maxlen=SEQ_FRAMES_OVERRIDE)}
    gest._open_buf = {"Left": deque(maxlen=SEQ_FRAMES_OVERRIDE), "Right": deque(maxlen=SEQ_FRAMES_OVERRIDE)}
    gc.SEQ_WINDOW = SEQ_WINDOW_OVERRIDE
    fine = gc.FineModeTracker()  # 主要用于保留原 draw_panel 的完整 UI
    state = gc.VirtualState()

    active_target = None
    control_until = 0.0
    last_kinect_recheck = 0.0
    last_control_action_time = 0.0
    # 只记录真实控制动作发生的时间，不包括刚进入控制会话或目标切换
    last_real_action_time = 0.0

    session_base_b = None
    session_base_v = None
    session_base_t = None
    lc_was_active = False
    rc_was_active = False
    last_log_b = 0.0
    last_log_v = 0.0
    prev_x_axis_trigger = {"Left": None, "Right": None}

    frame_idx = 0
    cached_gr_result = {"Left": None, "Right": None}
    hand_stable = {"Left": 0, "Right": 0}
    hand_lost_ct = {"Left": 0, "Right": 0}
    gesture_stable = {"Left": 0, "Right": 0}
    gesture_lost_ct = {"Left": 0, "Right": 0}

    # 用于 OPEN <-> FIST 转换触发 toggle。
    # last_toggle_gesture 记录上一帧/上一段稳定手势；
    # last_toggle_time 防止同一个转换被连续触发多次。
    last_toggle_gesture = {"Left": None, "Right": None}
    last_toggle_time = {"Left": 0.0, "Right": 0.0}
    # 记录当前这段 open/fist 动作起始时间，用于 2 秒内完成转换判断
    last_toggle_gesture_time = {"Left": 0.0, "Right": 0.0}

    panel_cache = None
    last_roi = None
    roi_keep = 0

    PL = mp_pose.PoseLandmark
    IDX = {
        "LS": PL.LEFT_SHOULDER, "LE": PL.LEFT_ELBOW, "LW": PL.LEFT_WRIST,
        "RS": PL.RIGHT_SHOULDER, "RE": PL.RIGHT_ELBOW, "RW": PL.RIGHT_WRIST,
        "LH": PL.LEFT_HIP, "RH": PL.RIGHT_HIP,
    }

    print("=" * 78)
    print("Intent-driven Mijia Control")
    print("Idle mode   : Kinect intent only")
    print("Control mode: Mediapipe control + Kinect intent refresh")
    print(f"control_session_sec = {CONTROL_SESSION_SEC}  # 0.0 = no fixed 2s lock")
    print(f"control_end_sec       = {CONTROL_END_SEC}  # 0.0 = no control-end protection")
    print(f"action_switch_protect = {ACTION_SWITCH_PROTECT_SEC}  # 0.0 = disabled")
    print(f"control_idle_exit_sec = {CONTROL_IDLE_EXIT_SEC}")
    print(f"kinect_recheck_sec = {KINECT_RECHECK_SEC}  # 0.0 = realtime after control end")
    print(f"gesture_recognizer_interval = every {GR_EVERY_N_FRAMES} frames")
    print("Press q to quit")
    print("=" * 78)

    while True:
        frame_idx += 1

        frame_data = kinect_runtime.get_live_frame_data()
        if frame_data is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame_data["color_image"], 1)
        joints_xyz = frame_data["joints_xyz"]
        has_body = frame_data["has_body"]
        now = time.time()

        # =========================================================
        # 1) Kinect intent：
        #    - Idle 状态：正常实时检测，用于进入控制会话
        #    - Control 状态：不再等待控制结束保护，realtime intent 每帧检测下一目标
        # =========================================================
        result = None
        should_check_intent = False
        control_finished = True  # 无控制结束保护，始终允许 realtime intent 判断目标

        if joints_xyz is not None:
            if active_target is None:
                # 空闲状态下，持续运行 realtime intent
                should_check_intent = True
            else:
                # 控制状态下也实时运行 intent。
                # KINECT_RECHECK_SEC = 0.0 表示每帧检测；若以后想限频，可把它改成 0.1/0.2。
                if KINECT_RECHECK_SEC <= 0 or now - last_kinect_recheck >= KINECT_RECHECK_SEC:
                    should_check_intent = True

        if should_check_intent:
            try:
                result = predictor.update(joints_xyz)
                last_kinect_recheck = now
            except Exception as e:
                print(f"[ERROR] predictor.update failed: {e}")

        # Idle 状态下，只有 display_pred 真正确认后才进入控制会话
        target_pred = get_target_pred_from_result(result, prefer_key="stable_pred")
        if active_target is None and target_pred in (1, 2):
            active_target = int(target_pred)
            control_until = now + CONTROL_SESSION_SEC
            last_control_action_time = now
            last_kinect_recheck = now

            pose = mp_pose.Pose(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                model_complexity=0,
            )
            hands = mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=HANDS_DET_CONF,
                min_tracking_confidence=HANDS_TRK_CONF,
            )
            gesture_recognizer = load_gesture_recognizer()

            # 重置会话相关状态，避免上一轮残留
            session_base_b = None
            session_base_v = None
            session_base_t = None
            lc_was_active = False
            rc_was_active = False
            prev_x_axis_trigger = {"Left": None, "Right": None}
            cached_gr_result = {"Left": None, "Right": None}
            hand_stable = {"Left": 0, "Right": 0}
            hand_lost_ct = {"Left": 0, "Right": 0}
            gesture_stable = {"Left": 0, "Right": 0}
            gesture_lost_ct = {"Left": 0, "Right": 0}
            last_toggle_gesture = {"Left": None, "Right": None}
            last_toggle_time = {"Left": 0.0, "Right": 0.0}
            last_toggle_gesture_time = {"Left": 0.0, "Right": 0.0}
            last_roi = None
            roi_keep = 0

            print(f"✅ Enter control session for Device{active_target}")

        # 控制中不再使用 0.8s 控制结束保护，允许根据 realtime intent 实时切换 active_target
        elif active_target is not None and target_pred in (1, 2):
            new_target = int(target_pred)

            # 无固定 2 秒锁死、无 0.8 秒控制结束保护、无动作后切换保护：
            # 只要 stable_pred 输出另一个目标，就立即切换 active_target。
            if new_target != active_target:
                active_target = new_target
                control_until = now + CONTROL_SESSION_SEC
                # 切换目标只刷新会话存活时间，不算真实控制动作
                last_control_action_time = now
                last_kinect_recheck = now
                session_base_b = state.brightness
                session_base_v = state.volume
                session_base_t = state.temp
                prev_x_axis_trigger = {"Left": None, "Right": None}
                print(f"🔁 Switch control target to Device{active_target}")

        # =========================================================
        # 2) Control mode: 跑 Mediapipe 控制，同时 Kinect intent 仍可实时刷新 active_target
        # =========================================================
        palm_pts = {"Left": None, "Right": None}
        hand_visible = {"Left": 0, "Right": 0}

        # 不要每帧直接把 hand_ready 置 False。
        # 如果上一段时间内这只手稳定出现过，并且当前丢失帧数仍在容错范围内，
        # 就继续认为它 ready，避免 MediaPipe 短暂丢点导致控制立刻断裂。
        hand_ready = {
            "Left": hand_stable.get("Left", 0) >= 1 and hand_lost_ct.get("Left", 0) <= HAND_LOST_GRACE_OVERRIDE,
            "Right": hand_stable.get("Right", 0) >= 1 and hand_lost_ct.get("Right", 0) <= HAND_LOST_GRACE_OVERRIDE,
        }

        pose_res = None
        hands_res = None
        gr_result = dict(cached_gr_result)

        roi = None
        resize_info = None
        if active_target is not None and pose is not None and hands is not None:
            # Pose 用半分辨率找 ROI，再在放大 ROI 上跑 Hands / GestureRecognizer
            small = cv2.resize(frame, None, fx=POSE_SCALE, fy=POSE_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False
            pose_res = pose.process(rgb_small)
            rgb_small.flags.writeable = True

            roi = get_pose_roi_from_small(frame.shape, pose_res.pose_landmarks if pose_res else None, POSE_SCALE, last_roi=last_roi)
            if roi is not None:
                last_roi = roi
                roi_keep = ROI_FALLBACK_KEEP
            elif last_roi is not None and roi_keep > 0:
                roi = last_roi
                roi_keep -= 1

            if roi is not None:
                if SHOW_ROI_BOX:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                roi_bgr, resize_info = crop_and_resize_roi(frame, roi)
                if roi_bgr is not None:
                    rgb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                    rgb_roi.flags.writeable = False
                    hands_res = hands.process(rgb_roi)
                    rgb_roi.flags.writeable = True

                    # 如果 ROI 内没有检测到手，定期用全图 Hands 兜底。
                    # 这主要解决“手放在身前，但 Pose-based ROI 没包住手”的情况。
                    used_fullframe_fallback = False
                    if (
                        (hands_res is None or not hands_res.multi_hand_landmarks)
                        and FULLFRAME_HAND_FALLBACK_EVERY > 0
                        and frame_idx % FULLFRAME_HAND_FALLBACK_EVERY == 0
                    ):
                        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_full.flags.writeable = False
                        fallback_res = hands.process(rgb_full)
                        rgb_full.flags.writeable = True

                        if fallback_res is not None and fallback_res.multi_hand_landmarks:
                            hands_res = fallback_res
                            roi = (0, 0, frame.shape[1], frame.shape[0])
                            resize_info = fullframe_resize_info(frame)
                            used_fullframe_fallback = True

                    # GestureRecognizer 隔帧跑，减轻卡顿。
                    # 如果使用了全图兜底，就在全图上做 GestureRecognizer；
                    # 否则仍然在放大 ROI 上做 GestureRecognizer。
                    if gesture_recognizer is not None and frame_idx % GR_EVERY_N_FRAMES == 0:
                        try:
                            if used_fullframe_fallback:
                                gr_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            else:
                                gr_rgb = rgb_roi

                            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=gr_rgb)
                            gr_out = gesture_recognizer.recognize(mp_img)
                            gr_result = {"Left": None, "Right": None}
                            for g_list, h_list in zip(gr_out.gestures, gr_out.handedness):
                                side = h_list[0].display_name
                                cat_name = g_list[0].category_name
                                gr_result[side] = cat_name
                            cached_gr_result = dict(gr_result)
                        except Exception as e:
                            print(f"[WARN] gesture_recognizer failed: {e}")

        # Hands landmarks + 手势 + fine mode
        detected_sides = set()
        if active_target is not None and hands_res is not None and hands_res.multi_hand_landmarks and roi is not None and resize_info is not None:
            PALM_BASE = [0, 5, 9, 13, 17]
            for hlm, hinfo in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                label = hinfo.classification[0].label
                detected_sides.add(label)

                remapped = remap_hand_landmarks_to_full(hlm, roi, resize_info, frame.shape)

                palm_x = float(np.mean([remapped[i][0] for i in PALM_BASE]))
                palm_y = float(np.mean([remapped[i][1] for i in PALM_BASE]))
                palm_pts[label] = np.array([palm_x, palm_y])

                hand_visible[label] = sum(
                    1 for x, y in remapped
                    if 0.0 < x < 1.0 and 0.0 < y < 1.0
                )

                if hand_visible[label] >= HAND_READY_MIN_VISIBLE:
                    hand_lost_ct[label] = 0
                    hand_stable[label] = min(hand_stable[label] + 1, gc.HAND_STABLE_FRAMES * 3)
                else:
                    hand_lost_ct[label] += 1
                    if hand_lost_ct[label] > HAND_LOST_GRACE_OVERRIDE:
                        hand_stable[label] = 0
                hand_ready[label] = hand_stable[label] >= 1

                if hand_visible[label] >= GESTURE_MIN_VISIBLE:
                    gesture_lost_ct[label] = 0
                    gesture_stable[label] = min(gesture_stable[label] + 1, GESTURE_STABLE_FRAMES * 3)
                else:
                    gesture_lost_ct[label] += 1
                    if gesture_lost_ct[label] > GESTURE_LOST_GRACE:
                        gesture_stable[label] = 0
                gesture_ready = gesture_stable[label] >= GESTURE_STABLE_FRAMES

                # fine mode / gesture 逻辑继续使用 ROI 放大后的 hlm
                fine.update(label, hlm)

                geom_open, geom_fist, _dbg = gc.classify_open_fist_nofacing(hlm)
                side_open, side_fist, side_ext = classify_open_fist_side_fallback(hlm)
                gr_name = gr_result.get(label)
                mp_open = gr_name == "Open_Palm"
                mp_fist = gr_name == "Closed_Fist"

                # 识别优先级：
                # 1) 原 no-facing 几何规则：正面/常规角度更稳定
                # 2) MediaPipe GestureRecognizer：对标准 Open/Fist 有帮助
                # 3) 侧面兜底分类器：专门补侧边手掌/拳头难识别的问题
                if geom_open and not geom_fist:
                    open_hand, fist = True, False
                elif geom_fist and not geom_open:
                    open_hand, fist = False, True
                elif USE_GESTURE_RECOGNIZER_AS_FALLBACK and gesture_ready and mp_open and not mp_fist:
                    open_hand, fist = True, False
                elif USE_GESTURE_RECOGNIZER_AS_FALLBACK and gesture_ready and mp_fist and not mp_open:
                    open_hand, fist = False, True
                elif USE_SIDE_OPEN_FIST_FALLBACK and side_open and not side_fist:
                    open_hand, fist = True, False
                elif USE_SIDE_OPEN_FIST_FALLBACK and side_fist and not side_open:
                    open_hand, fist = False, True
                else:
                    open_hand, fist = False, False

                # =====================================================
                # Open/Fist 开关逻辑
                # =====================================================
                if USE_TRANSITION_TOGGLE_SWITCH:
                    # 方向式转换逻辑：
                    # FIST -> OPEN = 开
                    # OPEN -> FIST = 关
                    #
                    # 关键点：
                    # 1) current_gesture 只有在稳定识别到 OPEN/FIST 时才更新。
                    # 2) 中间如果短暂 UNKNOWN，不清空 last_toggle_gesture，
                    #    因此允许用户在 3 秒窗口内完成转换。
                    # 3) 超过 3 秒才转换，则不触发，只把当前手势作为新的起点。
                    current_gesture = None
                    if gesture_ready and open_hand and not fist:
                        current_gesture = "open"
                    elif gesture_ready and fist and not open_hand:
                        current_gesture = "fist"

                    if current_gesture is not None:
                        prev_gesture = last_toggle_gesture.get(label)
                        prev_time = last_toggle_gesture_time.get(label, 0.0)

                        # 第一次看到稳定 OPEN/FIST：只记录起点，不触发。
                        if prev_gesture not in ("open", "fist"):
                            last_toggle_gesture[label] = current_gesture
                            last_toggle_gesture_time[label] = now

                        # 同一手势持续出现：不反复刷新起点时间。
                        # 这样用户维持 OPEN 之后，仍然需要在 3 秒内转成 FIST 才算完成动作。
                        elif prev_gesture == current_gesture:
                            pass

                        else:
                            # 出现 OPEN <-> FIST 转换：
                            # 只有在 3 秒动作窗口内完成，才允许执行方向式开关。
                            within_transition_window = (
                                prev_time > 0.0
                                and now - prev_time <= GESTURE_TRANSITION_WINDOW_SEC
                            )
                            cooldown_ok = (
                                now - last_toggle_time.get(label, 0.0) >= GESTURE_TOGGLE_COOLDOWN_SEC
                            )

                            if within_transition_window and cooldown_ok and active_target is not None:
                                # FIST -> OPEN = 开；OPEN -> FIST = 关
                                if prev_gesture == "fist" and current_gesture == "open":
                                    target_on = True
                                elif prev_gesture == "open" and current_gesture == "fist":
                                    target_on = False
                                else:
                                    target_on = None

                                if target_on is not None:
                                    did_action = set_active_device_switch(active_target, target_on, state, worker)
                                    if did_action:
                                        control_until = now + CONTROL_SESSION_SEC
                                        last_control_action_time = now
                                        last_real_action_time = now
                                        last_toggle_time[label] = now
                                        state.push_log(
                                            f"{label} {prev_gesture.upper()}->{current_gesture.upper()} "
                                            f"{'ON' if target_on else 'OFF'}"
                                        )

                                # 动作完成后清空起点，必须重新做一组转换才会再次触发。
                                last_toggle_gesture[label] = None
                                last_toggle_gesture_time[label] = 0.0

                            else:
                                # 超时或冷却未结束：
                                # 不执行开关，把当前稳定手势作为新起点。
                                last_toggle_gesture[label] = current_gesture
                                last_toggle_gesture_time[label] = now

                    # current_gesture is None 表示 UNKNOWN：
                    # 不清空 last_toggle_gesture，允许中间短暂丢失/UNKNOWN 后继续完成转换。

                else:
                    # 旧逻辑：
                    # 仍然使用 GestureTracker 的 on/off 输出，可通过 INVERT_OPEN_FIST_ACTION 反转。
                    action = None
                    if gesture_ready and (open_hand ^ fist):
                        old_hold = gc.GESTURE_HOLD_SEC
                        gc.GESTURE_HOLD_SEC = GESTURE_HOLD_SEC_OVERRIDE
                        try:
                            action = gest.update(label, fist, open_hand)
                        finally:
                            gc.GESTURE_HOLD_SEC = old_hold

                    if action and gest.can_trigger(label) and active_target is not None:
                        effective_action = action
                        if INVERT_OPEN_FIST_ACTION:
                            if action == "on":
                                effective_action = "off"
                            elif action == "off":
                                effective_action = "on"

                        if active_target == 1:
                            if effective_action == "on":
                                state.lamp_on = True
                                state.push_log("Lamp ON")
                                worker.send("lamp_on", True)
                            elif effective_action == "off":
                                state.lamp_on = False
                                state.push_log("Lamp OFF")
                                worker.send("lamp_on", False)

                        elif active_target == 2:
                            if effective_action == "on":
                                state.speaker_playing = True
                                state.push_log("Speaker PLAY")
                                worker.send("play", 1)
                            elif effective_action == "off":
                                state.speaker_playing = False
                                state.push_log("Speaker STOP")
                                worker.send("stop", 1)

                        control_until = now + CONTROL_SESSION_SEC
                        last_control_action_time = now
                        last_real_action_time = now
                        gest.mark(label)

                # UI 显示层：
                # 显示手掌/手指关键点，但隐藏 MediaPipe 手部骨骼连接线。
                # 识别逻辑仍然正常使用 hlm / remapped，不受显示开关影响。
                pts_px = np.array([(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in remapped], dtype=np.int32)

                if DRAW_HAND_CONNECTIONS:
                    for a, b in mp_hands.HAND_CONNECTIONS:
                        if a < len(pts_px) and b < len(pts_px):
                            cv2.line(frame, tuple(pts_px[a]), tuple(pts_px[b]), (0, 220, 120), 2)

                if DRAW_HAND_POINTS:
                    # 关键点区分：
                    # - 手掌基准点：腕部 + 四个掌指根部，用较大圆点
                    # - 指尖点：五个手指尖，用更明显圆点
                    # - 其他指节：普通小圆点
                    palm_ids = {0, 5, 9, 13, 17}
                    fingertip_ids = {4, 8, 12, 16, 20}

                    for i, p in enumerate(pts_px):
                        if i in fingertip_ids:
                            cv2.circle(frame, tuple(p), 5, (0, 180, 255), -1)
                        elif i in palm_ids:
                            cv2.circle(frame, tuple(p), 5, (0, 220, 120), -1)
                        else:
                            cv2.circle(frame, tuple(p), 3, (255, 220, 80), -1)

                if DRAW_GESTURE_TEXT:
                    if open_hand:
                        gesture_text = f"{label}: OPEN"
                        gesture_color = (0, 220, 120)
                    elif fist:
                        gesture_text = f"{label}: FIST"
                        gesture_color = (0, 180, 255)
                    else:
                        gesture_text = f"{label}: --"
                        gesture_color = (180, 180, 180)

                    tx = int(palm_x * frame.shape[1])
                    ty = int(palm_y * frame.shape[0])
                    cv2.rectangle(frame, (tx - 8, ty - 30), (tx + 120, ty - 5), (20, 20, 20), -1)
                    put(frame, gesture_text, tx, ty - 12, 0.52, gesture_color, 2)

        # 对当前帧没有检测到的手做丢失计数，但不立刻断控。
        # 这一步是减少 MediaPipe 短暂丢手的关键。
        if active_target is not None:
            for side in ("Left", "Right"):
                if side not in detected_sides:
                    hand_lost_ct[side] += 1
                    gesture_lost_ct[side] += 1

                    if hand_lost_ct[side] > HAND_LOST_GRACE_OVERRIDE:
                        hand_stable[side] = 0

                    if gesture_lost_ct[side] > GESTURE_LOST_GRACE:
                        gesture_stable[side] = 0

                    hand_ready[side] = (
                        hand_stable.get(side, 0) >= 1
                        and hand_lost_ct.get(side, 0) <= HAND_LOST_GRACE_OVERRIDE
                    )

        # Pose landmarks + 合并后的任意手控制
        if pose_res is not None and pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark

            def gn(k):
                return np.array([lm[IDX[k]].x, lm[IDX[k]].y])

            ls, le, lw = gn("LS"), gn("LE"), gn("LW")
            rs, re, rw = gn("RS"), gn("RE"), gn("RW")
            lh, rh = gn("LH"), gn("RH")

            # 仍然分别估计左右手轨迹，但功能上不再区分左右手
            lc.update(ls, le, lw, rs, lh, palm_pt=palm_pts["Left"])
            rc.update(rs, re, rw, ls, rh, palm_pt=palm_pts["Right"])

            if (lc.active and not lc_was_active) or (rc.active and not rc_was_active):
                session_base_b = state.brightness
                session_base_t = state.temp
                session_base_v = state.volume

            lc_was_active = lc.active
            rc_was_active = rc.active

            if active_target is not None:
                primary_ctrl, primary_side, primary_dt = choose_primary_control_hand(
                    lc, rc, hand_ready, hand_visible
                )

                if primary_ctrl is not None:
                    session_base_b, session_base_v, session_base_t, last_log_b, last_log_v, action_used = apply_control_for_any_hand(
                        primary_ctrl,
                        primary_side,
                        active_target,
                        state,
                        worker,
                        hand_ready,
                        hand_visible,
                        primary_dt,
                        session_base_b,
                        session_base_v,
                        session_base_t,
                        prev_x_axis_trigger,
                        now,
                        last_log_b,
                        last_log_v,
                    )

                    if action_used:
                        # 连续调节是真实控制动作：刷新控制动作时间；不再触发切换保护
                        control_until = now + CONTROL_SESSION_SEC
                        last_control_action_time = now
                        last_real_action_time = now


        # =========================================================
        # 3) 控制会话生命周期：无真实控制动作一段时间后释放 MediaPipe
        # =========================================================
        if active_target is not None and now - last_control_action_time > CONTROL_IDLE_EXIT_SEC:
            active_target = None
            control_until = 0.0
            last_kinect_recheck = 0.0
            last_real_action_time = 0.0

            if pose is not None:
                try:
                    pose.close()
                except Exception:
                    pass
            if hands is not None:
                try:
                    hands.close()
                except Exception:
                    pass

            pose = None
            hands = None
            gesture_recognizer = None

            # 重置控制链路状态，避免下一次进入时继承旧轨迹/手势缓存
            lc = gc.ArmController("left")
            rc = gc.ArmController("right")
            gest = gc.GestureTracker()
            gest._fist_buf = {"Left": deque(maxlen=SEQ_FRAMES_OVERRIDE), "Right": deque(maxlen=SEQ_FRAMES_OVERRIDE)}
            gest._open_buf = {"Left": deque(maxlen=SEQ_FRAMES_OVERRIDE), "Right": deque(maxlen=SEQ_FRAMES_OVERRIDE)}
            fine = gc.FineModeTracker()
            lc_was_active = False
            rc_was_active = False
            prev_x_axis_trigger = {"Left": None, "Right": None}
            cached_gr_result = {"Left": None, "Right": None}
            hand_stable = {"Left": 0, "Right": 0}
            hand_lost_ct = {"Left": 0, "Right": 0}
            gesture_stable = {"Left": 0, "Right": 0}
            gesture_lost_ct = {"Left": 0, "Right": 0}
            last_toggle_gesture = {"Left": None, "Right": None}
            last_toggle_time = {"Left": 0.0, "Right": 0.0}
            last_toggle_gesture_time = {"Left": 0.0, "Right": 0.0}
            last_roi = None
            roi_keep = 0

            print("💤 Exit control session: back to Kinect intent only")

        # =========================================================
        # 4) UI
        # =========================================================
        gc.draw_panel(
            frame=frame,
            state=state,
            lc=lc,
            rc=rc,
            gest=gest,
            fine=fine,
            hand_visible=hand_visible,
            hand_ready=hand_ready,
        )

        draw_intent_status_box(
            frame=frame,
            result=result,
            active_target=active_target,
            has_body=has_body,
            mediapipe_on=(pose is not None and hands is not None),
            control_until=control_until,
            intent_recheck_sec=KINECT_RECHECK_SEC,
        )

        cv2.imshow("Intent-driven Mijia Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    if pose is not None:
        pose.close()
    if hands is not None:
        hands.close()
    kinect_runtime.close()


def select_device_by_keyword(devices, keyword: str, role_name: str):
    keyword = keyword.strip()
    matches = [d for d in devices if keyword and keyword in d.get("name", "")]

    if len(matches) == 1:
        chosen = matches[0]
        print(f"✅ {role_name} 自动匹配到: {chosen['name']}  ({chosen['model']})")
        return chosen

    if len(matches) > 1:
        print(f"\n⚠️ {role_name} 关键词 '{keyword}' 匹配到多个设备，请手动选一个：")
        for i, d in enumerate(matches):
            print(f"  [{i}] {d['name']}  ({d['model']})")
        idx = int(input("请输入匹配结果编号: ").strip())
        return matches[idx]

    print(f"\n⚠️ 没有找到名字包含 '{keyword}' 的{role_name}，改为从全部设备里手动选择：")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d['name']}  ({d['model']})")
    idx = int(input("请输入设备编号: ").strip())
    return devices[idx]


def main():
    print("🚀 启动 Intent-driven 米家控制器")
    api = gc.get_api()

    devices = api.get_devices_list()
    print("📱 检测到以下设备：")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d['name']}  ({d['model']})")

    if len(devices) < 2:
        raise RuntimeError("设备数量不足，至少需要 2 个设备")

    # 按你的固定默认逻辑：
    # Device2 = devices[0]（台灯）
    # Device1 = devices[1]（音箱）
    lamp_info = devices[0]      # Device2
    speaker_info = devices[1]   # Device1

    print(f"✅ 默认 Device1(音箱) -> [1] {speaker_info['name']}")
    print(f"✅ 默认 Device2(台灯) -> [0] {lamp_info['name']}")

    lamp_dev = gc.mijiaDevice(api, dev_name=lamp_info["name"], sleep_time=0.1)
    speaker_dev = gc.mijiaDevice(api, dev_name=speaker_info["name"], sleep_time=0.1)
    worker = gc.MijiaWorker(lamp_dev, speaker_dev)

    model_path = resolve_model_path()
    device_json_path = resolve_device_json_path()

    print("=" * 78)
    print("Realtime attention parameters")
    print(f"model_path            = {model_path}")
    print(f"device_json_path      = {device_json_path}")
    print(f"class_conf_thresh     = {DEFAULT_CLASS_CONF_THRESH}")
    print(f"class_switch_patience = {DEFAULT_CLASS_SWITCH_PATIENCE}")
    print(f"hold_by_state         = {DEFAULT_HOLD_BY_STATE}")
    print(f"logit_bias            = {DEFAULT_LOGIT_BIAS}")
    print(f"shoulder_overlap_range_m = {DEFAULT_SHOULDER_OVERLAP_RANGE_M}")
    print(f"arm_boost_min/max     = ({DEFAULT_ARM_BOOST_MIN}, {DEFAULT_ARM_BOOST_MAX})")
    print("ema_alpha             = 0.5")
    print("use_ema               = True")
    print("prob_window           = 5")
    print(f"kinect_recheck_sec    = {KINECT_RECHECK_SEC}  # 0.0 = realtime after control end")
    print(f"control_end_sec       = {CONTROL_END_SEC}  # 0.0 = no control-end protection")
    print(f"action_switch_protect = {ACTION_SWITCH_PROTECT_SEC}  # 0.0 = disabled")
    print(f"control_idle_exit_sec = {CONTROL_IDLE_EXIT_SEC}")
    print(f"roi_target_size       = {ROI_TARGET_SIZE}")
    print(f"roi_margin_x/top/btm  = ({ROI_MARGIN_X}, {ROI_MARGIN_TOP}, {ROI_MARGIN_BOTTOM})")
    print(f"show_roi_box          = {SHOW_ROI_BOX}")
    print(f"fullframe_fallback    = every {FULLFRAME_HAND_FALLBACK_EVERY} frames")
    print(f"hands_det/trk_conf    = ({HANDS_DET_CONF}, {HANDS_TRK_CONF})")
    print(f"hand_ready_min_visible= {HAND_READY_MIN_VISIBLE}")
    print(f"hand_lost_grace       = {HAND_LOST_GRACE_OVERRIDE}")
    print(f"gesture_lost_grace    = {GESTURE_LOST_GRACE}")
    print(f"gr_every_n_frames     = {GR_EVERY_N_FRAMES}")
    print(f"gr_as_fallback        = {USE_GESTURE_RECOGNIZER_AS_FALLBACK}")
    print(f"invert_open_fist     = {INVERT_OPEN_FIST_ACTION}")
    print(f"side_fallback        = {USE_SIDE_OPEN_FIST_FALLBACK}")
    print(f"transition_toggle   = {USE_TRANSITION_TOGGLE_SWITCH}")
    print(f"directional_switch  = {USE_DIRECTIONAL_OPEN_FIST_SWITCH}")
    print(f"toggle_cooldown_sec = {GESTURE_TOGGLE_COOLDOWN_SEC}")
    print(f"transition_window  = {GESTURE_TRANSITION_WINDOW_SEC}")
    print(f"draw_hand_connections= {DRAW_HAND_CONNECTIONS}")
    print(f"draw_hand_points     = {DRAW_HAND_POINTS}")
    print(f"draw_gesture_text     = {DRAW_GESTURE_TEXT}")
    print("target_switch_pred    = stable_pred")
    print("=" * 78)

    predictor = RealtimeAttentionPredictor(
        model_path=model_path,
        device_json_path=device_json_path,
        ema_alpha=0.5,
        use_ema=True,
        prob_window=5,
        class_conf_thresh=DEFAULT_CLASS_CONF_THRESH,
        class_switch_patience=DEFAULT_CLASS_SWITCH_PATIENCE,
        hold_by_state=DEFAULT_HOLD_BY_STATE,
        logit_bias=DEFAULT_LOGIT_BIAS,
        shoulder_overlap_range_m=DEFAULT_SHOULDER_OVERLAP_RANGE_M,
        arm_boost_min=DEFAULT_ARM_BOOST_MIN,
        arm_boost_max=DEFAULT_ARM_BOOST_MAX,
    )

    print("=" * 78)
    print("Loaded realtime predictor")
    print(f"window_size           = {predictor.window_size}")
    print(f"stride                = {predictor.stride}")
    print(f"runtime_device        = {predictor.device}")
    print(f"model_family          = {getattr(predictor, 'model_family', 'unknown')}")
    print("loaded model kwargs:")
    for k, v in predictor.model_kwargs.items():
        print(f"  {k:<22} = {v}")
    print("=" * 78)

    kinect_runtime = KinectRuntime()
    run(worker, predictor, kinect_runtime)


if __name__ == "__main__":
    main()