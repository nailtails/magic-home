import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json
import os
from collections import deque
from filterpy.kalman import KalmanFilter
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mijiaAPI import mijiaAPI, mijiaLogin, mijiaDevice

# ── 意图模型（可选，文件不存在时自动跳过）──────────────
_INTENT_MODEL    = None
_INTENT_STATS    = None
_INTENT_FEATURES = ["amp_y", "vel_y", "delta_y", "delta_x", "weight"]
INTENT_WINDOW    = 20       # 与训练时一致
INTENT_THRESH    = 0.35     # 置信度阈值（EMA平滑后，适当放宽）

def _load_intent_model(model_path=None, stats_path=None):
    global _INTENT_MODEL, _INTENT_STATS
    # 自动找脚本所在目录，不依赖运行时工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "intent_model.onnx")
    if stats_path is None:
        stats_path = os.path.join(script_dir, "intent_model_stats.json")
    global _INTENT_MODEL, _INTENT_STATS
    try:
        import onnxruntime as ort
        if os.path.exists(model_path) and os.path.exists(stats_path):
            _INTENT_MODEL = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            with open(stats_path) as f:
                _INTENT_STATS = json.load(f)
            print(f"✅ 意图模型已加载: {model_path}")
        else:
            print("⚠️  未找到意图模型文件，跳过意图过滤（放手保护仍然生效）")
    except ImportError:
        print("⚠️  onnxruntime 未安装，跳过意图过滤")
        print("   安装命令：pip install onnxruntime")

#  参数配置
# ── 激活区（手腕相对胸部代理点，除以肩宽归一化）──────
# 控制参考点上提一档：0.35×肩膀 + 0.65×髋关节，介于上腹与胯骨之间
# 比上一版更高一点，但仍明显低于胸口
ACTIVATE_THRESH   = 0.10    # amp_y > 0.10 → 进入控制模式
DEACTIVATE_THRESH = -0.12   # amp_y < -0.12 → 退出控制模式

# ── 运动轨迹控制（替换静态分区）────────────────────
MOTION_WIN        = 5        # 位移计算的帧窗口（5帧）
MOTION_DEAD_ZONE  = 0.018   # 低于此归一化位移不响应（过滤抖动）
MOTION_SENSITIVITY= 85.0    # 位移→数值变化的灵敏度系数（y轴）
MOTION_SENSITIVITY_X = 50.0 # x轴单独灵敏度提高：横向调节更明显
LEFT_SENSITIVITY_COMP = 1.3  # 左手灵敏度补偿系数
MOTION_K          = 4.0     # 抛物线开口系数（越大速率衰减越快）
MOTION_MIN_RATE   = 0.10    # 抛物线最低速率（边缘区域保留10%）
MOTION_PEAK       = 0.25    # 抛物线顶点高度（手掌高于胸口约0.25肩宽时速率最高）
AXIS_DOM_RATIO    = 1.20    # 主轴需比另一轴明显更大，减少横向移动时误触音量

# ── 轴锁定 ────────────────────────────────────────────
AXIS_LOCK_ENTER   = 4        # 30FPS 下更灵敏：连续4帧同轴就锁定
AXIS_LOCK_EXIT    = 6        # 30FPS 下更自然：连续6帧异轴/反向后解锁

# ── 放手保护 ─────────────────────────────────────────
DROP_VEL_THRESH   = 0.055   # 放手保护速度阈值（放宽，意图模型是主要过滤层）
DROP_STREAK       = 4        # 连续下落帧数（实际用 +2 = 6帧）
DEACTIVATE_FREEZE = 3        # 退出激活前冻结帧数（从5降到3）

# ── EMA平滑 ──────────────────────────────────────────
EMA_ALPHA = 0.12            # 越小越平滑

# ── 死区 + 速率限制 ──────────────────────────────────
DEAD_ZONE_B  = 1.2          # 亮度死区 %
DEAD_ZONE_V  = 0.3          # 音量死区 %
DEAD_ZONE_T  = 1.5          # 色温死区缩小，横向调节更容易生效
MAX_RATE_B   = 25.0         # 亮度最大速率 %/s
MAX_RATE_V   = 25.0         # 音量最大速率 %/s
MAX_RATE_T   = 800.0        # 色温最大速率 K/s

# ── 设备范围 ─────────────────────────────────────────
B_MIN, B_MAX = 1,    100
T_MIN, T_MAX = 2700, 6500

# ── 手势 ─────────────────────────────────────────────
GESTURE_COOLDOWN  = 1.2
SEQ_FRAMES        = 3        # 更灵敏：连续3帧确认
SEQ_WINDOW        = 2.0      # 序列完成最大时间窗口（秒），超过则重置
GESTURE_HOLD_SEC  = 0.12     # 单个 open/fist 先稳定约120ms，再进入序列缓冲

# ── 单次激活调节上限 ──────────────────────────────────
# 每次激活期间累计变化不超过此值，防止误触导致大幅跳变
SESSION_CAP_B = 30.0   # 亮度单次最多变化 30%
SESSION_CAP_V = 25.0   # 音量单次最多变化 25%
SESSION_CAP_T = 800.0  # 色温单次最多变化 800K
# Hands 模型检测到的关键点数低于此值时，禁止该侧手臂控制
# MediaPipe Hands 共21个点，>=15 表示手掌基本完整可见
# ── 连续检测确认 ──────────────────────────────────────
# 连续检测到手部关键点达到此帧数才允许调节，中断则冻结
HAND_STABLE_FRAMES = 4    # 需要连续4帧稳定检测才允许输出
HAND_LOST_GRACE    = 3
# y轴（上下）要求更多点，x轴（左右）手侧转时关键点少，要求宽松
HAND_MIN_VISIBLE_Y = 10   # 降低阈值（原16），先确保能正常控制
HAND_MIN_VISIBLE_X = 7
OK_PINCH_DIST     = 0.07   # OK手势：拇指食指3D距离 < 0.07 判定为捏合成圈
PINCH_CONFIRM_SEC = 0.25   # OK手势稳定确认时间（延长防误触）
PINCH_GRACE_SEC   = 0.25    # 丢失宽限时间（缩短，松开更灵敏）
FINE_SENSITIVITY  = 25.0   # 精细模式灵敏度
FINE_DEAD_ZONE    = 0.005
FINE_WIN          = 5

# ═══════════════════════════════════════════════════════
#  模拟版本：无米家API，设备状态全部在内存中维护
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
#  设备状态（本地缓存）
# ═══════════════════════════════════════════════════════
class VirtualState:
    def __init__(self):
        self.lamp_on         = False
        self.brightness      = 50.0
        self.temp            = 4000.0
        self.speaker_playing = False
        self.volume          = 50.0
        self.track           = 1
        self.log             = deque(maxlen=6)
        self.last_action     = ""
        self.last_action_ts  = 0.0

    def push_log(self, msg):
        self.last_action    = msg
        self.last_action_ts = time.time()
        self.log.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")
        print(f"[ACTION] {msg}")

# ═══════════════════════════════════════════════════════
#  米家异步推送worker
#  主循环只往队列里塞命令，后台线程负责实际调用API
#  节流：同类命令0.3s内只取最新一条
# ═══════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════
#  米家异步Worker：主循环只塞命令，后台线程实际调用API
#  节流：同key最新值覆盖旧值，0.3s内只发一次
# ═══════════════════════════════════════════════════════
API_THROTTLE = 0.3

class MijiaWorker:
    def __init__(self, lamp_device: mijiaDevice, speaker_device: mijiaDevice):
        self.lamp    = lamp_device
        self.speaker = speaker_device
        self._pending = {}
        self._lock    = threading.Lock()
        self._t       = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        last_sent = {}
        while True:
            time.sleep(0.02)
            with self._lock:
                if not self._pending:
                    continue
                batch = dict(self._pending)
                self._pending.clear()

            now = time.time()
            for key, val in batch.items():
                prev_ts = last_sent.get(key, 0)
                if now - prev_ts < API_THROTTLE:
                    with self._lock:
                        if key not in self._pending:
                            self._pending[key] = val
                    continue
                last_sent[key] = now
                try:
                    if   key == "brightness": self.lamp.brightness        = int(val)
                    elif key == "temp":       self.lamp.color_temperature = int(val)
                    elif key == "lamp_on":    self.lamp.on                = bool(val)
                    elif key == "volume":     self.speaker.volume         = int(val)
                    elif key == "play":
                        self.speaker.run_action("execute-text-directive", _in=["播放音乐", True])
                    elif key == "stop":
                        self.speaker.run_action("execute-text-directive", _in=["停止播放", True])
                    elif key == "next_track":
                        self.speaker.run_action("execute-text-directive", _in=["下一首", True])
                    elif key == "prev_track":
                        self.speaker.run_action("execute-text-directive", _in=["上一首", True])
                except Exception as e:
                    print(f"[MIJIA ERROR] {key}={val} → {e}")

    def send(self, key, val):
        with self._lock:
            self._pending[key] = val

# ═══════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════
def clamp(v, lo, hi): return max(lo, min(hi, v))


def _infer_intent(feat_buf):
    """
    用意图模型推理最近 INTENT_WINDOW 帧是否为有意调节。
    返回 0~1 的置信度，>=INTENT_THRESH 认为是有意调节。
    模型未加载时返回 1.0（默认放行）。
    """
    if _INTENT_MODEL is None or _INTENT_STATS is None:
        return 1.0
    try:
        mean = np.array(_INTENT_STATS["mean"], dtype=np.float32)
        std  = np.array(_INTENT_STATS["std"],  dtype=np.float32)
        x    = np.array(list(feat_buf), dtype=np.float32)   # [20, 5]
        x    = (x - mean) / std
        x    = x[np.newaxis, ...]                            # [1, 20, 5]
        out  = _INTENT_MODEL.run(None, {"input": x})[0]
        return float(out[0])
    except Exception as e:
        return 1.0   # 推理出错时默认放行

def calc_angle(a, b, c):
    ba = a - b;  bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(clamp(cos_a, -1.0, 1.0))))

# ═══════════════════════════════════════════════════════
#  EMA滤波器
# ═══════════════════════════════════════════════════════
class EMA:
    def __init__(self, alpha=EMA_ALPHA, init=0.0):
        self.alpha = alpha
        self.v     = init
        self.ready = False

    def update(self, raw):
        if not self.ready:
            self.v = raw;  self.ready = True
        else:
            self.v = self.alpha * raw + (1 - self.alpha) * self.v
        return self.v

# ═══════════════════════════════════════════════════════
#  运动轨迹追踪器（替换 DirectionFSM）
# ═══════════════════════════════════════════════════════
class MotionTracker:
    """
    维护最近 MOTION_WIN 帧的手腕坐标队列。
    每帧输出：
      axis  : "y" | "x" | None   主运动轴（带锁定，防止跳轴）
      delta : float               归一化位移（已除以肩宽），正/负表示方向
      weight: float               抛物线速率权重 0~1（胸口=1，上下递减）
    """
    def __init__(self):
        self.buf_y = deque(maxlen=MOTION_WIN)
        self.buf_x = deque(maxlen=MOTION_WIN)
        self.axis   = None
        self.delta  = 0.0
        self.weight = 0.0

        # 轴锁定状态
        self._locked_axis   = None   # 当前锁定的轴
        self._lock_counter  = 0      # 同轴连续帧计数（用于入锁）
        self._exit_counter  = 0      # 反向/异轴连续帧计数（用于解锁）

    def update(self, amp_y, amp_x, shoulder_width):
        self.buf_y.append(amp_y)
        self.buf_x.append(amp_x)

        if len(self.buf_y) < MOTION_WIN:
            self.axis = None; self.delta = 0.0; self.weight = 0.0
            return

        dy = self.buf_y[-1] - self.buf_y[0]
        dx = self.buf_x[-1] - self.buf_x[0]

        # 当前帧的原始主轴判断
        if abs(dy) >= abs(dx) * AXIS_DOM_RATIO and abs(dy) > MOTION_DEAD_ZONE:
            raw_axis  = "y"
            raw_delta = dy
        elif abs(dx) > abs(dy) * AXIS_DOM_RATIO and abs(dx) > MOTION_DEAD_ZONE:
            raw_axis  = "x"
            raw_delta = dx
        else:
            raw_axis  = None
            raw_delta = 0.0

        # 轴锁定逻辑
        if self._locked_axis is None:
            # 未锁定：连续同轴达到阈值才锁定
            if raw_axis is not None and raw_axis == self._last_raw_axis():
                self._lock_counter += 1
            else:
                self._lock_counter = 1
            if self._lock_counter >= AXIS_LOCK_ENTER and raw_axis is not None:
                self._locked_axis  = raw_axis
                self._exit_counter = 0
            self.axis  = raw_axis
            self.delta = raw_delta
        else:
            # 已锁定：异轴/静止连续N帧才解锁
            if raw_axis != self._locked_axis:
                self._exit_counter += 1
            else:
                self._exit_counter = 0
            if self._exit_counter >= AXIS_LOCK_EXIT:
                self._locked_axis  = raw_axis
                self._lock_counter = AXIS_LOCK_ENTER  # 立即锁定新轴
                self._exit_counter = 0
            # 锁定期间强制使用锁定轴的数据
            if self._locked_axis == "y":
                self.axis  = "y" if abs(dy) > MOTION_DEAD_ZONE else None
                self.delta = dy
            else:
                self.axis  = "x" if abs(dx) > MOTION_DEAD_ZONE else None
                self.delta = dx

        self._raw_axis_prev = raw_axis

        # 抛物线速率权重：以 MOTION_PEAK 为顶点，上下对称递减
        raw_w = 1.0 - MOTION_K * ((amp_y - MOTION_PEAK) ** 2)
        self.weight = max(MOTION_MIN_RATE, min(1.0, raw_w))

    def _last_raw_axis(self):
        return getattr(self, '_raw_axis_prev', None)

    def reset(self):
        self.buf_y.clear(); self.buf_x.clear()
        self.axis   = None; self.delta = 0.0; self.weight = 0.0
        self._locked_axis  = None
        self._lock_counter = 0
        self._exit_counter = 0
        self._raw_axis_prev = None

# ═══════════════════════════════════════════════════════
#  单侧手臂控制器
# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
#  卡尔曼滤波器（替换 EMA，2D 手腕坐标专用）
# ═══════════════════════════════════════════════════════
class KalmanWristFilter:
    """
    2D 卡尔曼滤波，状态向量 [x, y, vx, vy]
    专为归一化手腕坐标 (x,y) ∈ [0,1] 设计
    顺带输出速度，不需要再做帧差
    """
    def __init__(self, dt=1/30, process_noise=8e-4, obs_noise=8e-3):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # 状态转移：匀速运动模型
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1,  0, dt],
                              [0, 0,  1,  0],
                              [0, 0,  0,  1]], dtype=float)

        # 观测矩阵：只观测位置
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=float)

        self.kf.P = np.eye(4) * 0.1
        self.kf.R = np.eye(2) * obs_noise
        self.kf.Q = np.eye(4) * process_noise

        self._initialized = False

    def update(self, x: float, y: float):
        z = np.array([x, y])
        if not self._initialized:
            self.kf.x = np.array([x, y, 0.0, 0.0])
            self._initialized = True
            return x, y
        self.kf.predict()
        self.kf.update(z)
        return float(self.kf.x[0]), float(self.kf.x[1])

    @property
    def velocity(self):
        """直接读速度，无需帧差"""
        return float(self.kf.x[2]), float(self.kf.x[3])

    def reset(self):
        self._initialized = False
        self.kf.P = np.eye(4) * 0.1


# ═══════════════════════════════════════════════════════
#  单侧手臂控制器
# ═══════════════════════════════════════════════════════
class ArmController:
    def __init__(self, side: str):
        self.side   = side
        self.active = False

        # 卡尔曼滤波器替换 EMA（坐标平滑 + 速度直接输出）
        self.kf_wrist  = KalmanWristFilter(dt=1/30)
        self.angle_ema = EMA(alpha=0.08)   # 角度仍用EMA，变化慢无需卡尔曼

        self.amp_y          = 0.0
        self.amp_x          = 0.0
        self.angle          = 0.0
        self.shoulder_width = 0.18

        self.tracker   = MotionTracker()
        self.last_time = time.time()

        # 放手保护
        self._drop_streak    = 0
        self.frozen          = False
        self._deact_countdown = 0   # 退出激活倒计时，期间强制冻结

        # 意图模型特征缓冲
        self._intent_buf       = deque(maxlen=INTENT_WINDOW)
        self.intent_score      = 1.0
        self._intent_score_raw = 1.0
        self._intent_low_since = None   # 分数持续低于阈值的起始时间
        self._warmup           = 0

        # 速度（直接从卡尔曼读，不再帧差）
        self.vel_y = 0.0

    def update(self, shoulder, elbow, wrist, opp_shoulder, hip, palm_pt=None):
        sw = abs(shoulder[0] - opp_shoulder[0])
        if sw > 0.05:
            self.shoulder_width = sw
        sw = max(self.shoulder_width, 0.05)

        chest = 0.35 * shoulder + 0.65 * hip

        # 控制点：优先用手掌五点均值，没有则回退到肘腕中点
        if palm_pt is not None:
            ctrl_x, ctrl_y = palm_pt[0], palm_pt[1]
        else:
            ctrl_x = (wrist[0] + elbow[0]) / 2
            ctrl_y = (wrist[1] + elbow[1]) / 2

        raw_y = (chest[1] - ctrl_y) / sw
        raw_x_base = (ctrl_x - shoulder[0]) / sw
        raw_x = raw_x_base if self.side == "right" else -raw_x_base

        self.angle = self.angle_ema.update(calc_angle(shoulder, elbow, wrist))

        just_activated = False

        # 激活区判断（带滞后）
        # palm_pt 为 None 表示 Hands 模型没检测到这只手，禁止激活
        if not self.active:
            self.amp_y = raw_y
            self.amp_x = raw_x
            self.vel_y = 0.0
            if self.amp_y > ACTIVATE_THRESH and palm_pt is not None:
                self.active    = True
                just_activated = True
                self.tracker.reset()
                self.kf_wrist.reset()
                self.kf_wrist.update(ctrl_x, ctrl_y)
                self.frozen           = False
                self._drop_streak     = 0
                self._intent_buf.clear()
                self._warmup          = MOTION_WIN   # 激活后前N帧禁止输出，等buf填满稳定值
        else:
            if raw_y < DEACTIVATE_THRESH:
                # 不立即退出，先冻结N帧再退出，防止放手瞬间带动数值
                self._deact_countdown += 1
                self.frozen = True
                if self._deact_countdown >= DEACTIVATE_FREEZE:
                    self.active           = False
                    self.tracker.reset()
                    self.frozen           = False
                    self._drop_streak     = 0
                    self._deact_countdown = 0
                    self.intent_score      = 1.0
                    self._intent_score_raw = 1.0
                    self._intent_low_since = None
                self.amp_y = raw_y
                self.amp_x = raw_x
                return self.active
            else:
                self._deact_countdown = 0

            # 激活状态：卡尔曼推进，输入与激活判断用同一个控制点
            fx, fy   = self.kf_wrist.update(ctrl_x, ctrl_y)
            _, kf_vy = self.kf_wrist.velocity
            self.amp_y = (chest[1] - fy) / sw
            raw_fx = (fx - shoulder[0]) / sw
            self.amp_x = raw_fx if self.side == "right" else -raw_fx
            self.vel_y = -kf_vy / sw

        # 运动追踪（仅激活且非刚激活帧）
        if self.active:
            self.tracker.update(self.amp_y, self.amp_x, sw)

            # 预热倒计时：激活后前 MOTION_WIN 帧等 buf 填满，不输出控制
            if self._warmup > 0:
                self._warmup -= 1

            # ── 意图特征入队 ──────────────────────────
            self._intent_buf.append([
                self.amp_y, self.vel_y,
                self.tracker.delta if self.tracker.axis == "y" else 0.0,
                self.tracker.delta if self.tracker.axis == "x" else 0.0,
                self.tracker.weight,
            ])
            if len(self._intent_buf) == INTENT_WINDOW:
                raw = _infer_intent(self._intent_buf)
                self._intent_score_raw = raw
                # 非对称 EMA：下降慢，上升快
                if raw < self.intent_score:
                    alpha = 0.15
                else:
                    alpha = 0.6
                self.intent_score = alpha * raw + (1 - alpha) * self.intent_score

            # 超时重置：intent_score 持续低于阈值超过 1.2 秒则强制重置
            # 防止 MediaPipe 短暂识别异常导致长时间卡死
            if self.intent_score < INTENT_THRESH:
                if self._intent_low_since is None:
                    self._intent_low_since = time.time()
                elif time.time() - self._intent_low_since > 1.2:
                    self.intent_score      = 1.0
                    self._intent_score_raw = 1.0
                    self._intent_low_since = None
                    self._intent_buf.clear()
            else:
                self._intent_low_since = None
            
            if not just_activated:
                # 放手保护已由意图模型全权负责，硬编码 frozen 完全移除
                self.frozen       = False
                self._drop_streak = 0
        else:
            self.tracker.reset()

        return self.active

    @property
    def axis(self):
        return self.tracker.axis

    @property
    def delta(self):
        return self.tracker.delta

    @property
    def weight(self):
        return self.tracker.weight

    def get_dt(self):
        now = time.time()
        dt  = clamp(now - self.last_time, 0.0, 0.1)
        self.last_time = now
        return dt

    @staticmethod
    def rate_limit(current, target, max_rate, dt):
        delta = target - current
        max_d = max_rate * dt
        return current + clamp(delta, -max_d, max_d)

    @staticmethod
    def dead_zone(current, target, dead):
        return target if abs(target - current) > dead else current

    def calc_delta_value(self):
        """每帧变化量 = 位移 × 灵敏度 × 抛物线权重 × 意图置信度"""
        if self.frozen or self.axis is None:
            return 0.0
        if self._warmup > 0:
            return 0.0   # 预热期：buf未填满，禁止输出防止初始跳变
        if self.intent_score < INTENT_THRESH:
            return 0.0   # 意图模型判断为无意动作，屏蔽输出
        sens = MOTION_SENSITIVITY if self.axis == "y" else MOTION_SENSITIVITY_X
        comp = LEFT_SENSITIVITY_COMP if self.side == "left" else 1.0
        return self.delta * sens * self.weight * comp

# ═══════════════════════════════════════════════════════
#  手势识别（no-facing 版本）
# ═══════════════════════════════════════════════════════
TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]

def _lm_xy(hlm, idx):
    p = hlm.landmark[idx]
    return np.array([p.x, p.y], dtype=np.float32)

def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

FINGERS = {
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}

def _finger_extended_score(hlm, mcp_i, pip_i, tip_i):
    wrist = _lm_xy(hlm, 0)
    mcp = _lm_xy(hlm, mcp_i)
    pip = _lm_xy(hlm, pip_i)
    tip = _lm_xy(hlm, tip_i)

    palm_size = max(_dist(_lm_xy(hlm, 0), _lm_xy(hlm, 9)), 1e-6)
    d_tip = _dist(tip, wrist) / palm_size
    d_pip = _dist(pip, wrist) / palm_size
    d_mcp = _dist(mcp, wrist) / palm_size

    extension_margin = d_tip - d_pip
    tip_from_base = d_tip - d_mcp
    return extension_margin, tip_from_base

def _thumb_open_score(hlm):
    wrist = _lm_xy(hlm, 0)
    thumb_tip = _lm_xy(hlm, 4)
    index_mcp = _lm_xy(hlm, 5)
    palm_size = max(_dist(wrist, index_mcp), 1e-6)
    thumb_spread = _dist(thumb_tip, index_mcp) / palm_size
    return thumb_spread

def classify_open_fist_nofacing(hlm):
    ext_count = 0
    curl_count = 0
    ext_margins = []
    tip_from_bases = []

    for _, (mcp_i, pip_i, tip_i) in FINGERS.items():
        ext_margin, tip_from_base = _finger_extended_score(hlm, mcp_i, pip_i, tip_i)
        ext_margins.append(ext_margin)
        tip_from_bases.append(tip_from_base)
        if ext_margin > 0.10 and tip_from_base > 0.35:
            ext_count += 1
        if ext_margin < 0.04 and tip_from_base < 0.28:
            curl_count += 1

    thumb_spread = _thumb_open_score(hlm)
    thumb_open = thumb_spread > 0.55
    thumb_closed = thumb_spread < 0.48

    avg_margin = float(np.mean(ext_margins))
    avg_tipbase = float(np.mean(tip_from_bases))

    raw_open = (ext_count >= 3 and avg_margin > 0.09 and avg_tipbase > 0.34)
    raw_fist = (curl_count >= 3 and avg_margin < 0.055 and avg_tipbase < 0.30)

    if raw_open and not thumb_open and ext_count == 3:
        raw_open = False
    if raw_fist and not thumb_closed and curl_count == 3:
        raw_fist = False

    dbg = {
        "ext_count": ext_count,
        "curl_count": curl_count,
        "avg_margin": round(avg_margin, 3),
        "avg_tipbase": round(avg_tipbase, 3),
        "thumb_spread": round(float(thumb_spread), 3),
    }
    return raw_open, raw_fist, dbg

def is_ok_gesture(lm):
    """
    OK 手势：拇指食指捏合成圈，食指弯曲，中/无名/小指三根伸直
      1. 拇指尖(4)与食指尖(8) 3D 距离 < 阈值
      2. 食指弯曲：指尖(8).y > PIP(6).y
      3. 中指/无名指/小指全部伸直：指尖.y < PIP.y
    """
    t = lm.landmark[4]; i = lm.landmark[8]
    pinch_dist = float(np.sqrt((t.x-i.x)**2 + (t.y-i.y)**2 + (t.z-i.z)**2))
    if pinch_dist >= OK_PINCH_DIST:
        return False
    if lm.landmark[8].y <= lm.landmark[6].y:
        return False
    return all(
        lm.landmark[tip].y < lm.landmark[pip].y
        for tip, pip in [(12,10), (16,14), (20,18)]
    )

class GestureTracker:
    """
    序列手势检测：
      开 = fist(5帧) → open(5帧)，严格顺序，2秒内完成
      关 = open(5帧) → fist(5帧)，严格顺序，2秒内完成
    中间出现其他状态立刻重置序列。
    """
    # 序列状态机的阶段
    _IDLE       = 0   # 等待第一个动作
    _GOT_FIRST  = 1   # 第一个动作已确认，等待第二个

    def __init__(self):
        # 每只手独立维护
        self._phase      = {"Left": self._IDLE,      "Right": self._IDLE}
        self._first_gest = {"Left": None,             "Right": None}   # "fist"/"open"
        self._seq_start  = {"Left": 0.0,              "Right": 0.0}
        self._fist_buf   = {"Left": deque(maxlen=SEQ_FRAMES),
                            "Right": deque(maxlen=SEQ_FRAMES)}
        self._open_buf   = {"Left": deque(maxlen=SEQ_FRAMES),
                            "Right": deque(maxlen=SEQ_FRAMES)}
        self.last_trigger= {"Left": 0.0,              "Right": 0.0}
        self._hold_start = {"Left": {"fist": None, "open": None},
                            "Right": {"fist": None, "open": None}}
        # 供UI显示
        self.pending     = {"Left": None,             "Right": None}   # "fist"/"open"/None

    def _confirmed(self, buf):
        return len(buf) == SEQ_FRAMES and all(buf)

    def update(self, label, fist, open_hand):
        now = time.time()
        if fist:
            if self._hold_start[label]["fist"] is None:
                self._hold_start[label]["fist"] = now
        else:
            self._hold_start[label]["fist"] = None

        if open_hand:
            if self._hold_start[label]["open"] is None:
                self._hold_start[label]["open"] = now
        else:
            self._hold_start[label]["open"] = None

        fist_ok = self._hold_start[label]["fist"] is not None and (now - self._hold_start[label]["fist"] >= GESTURE_HOLD_SEC)
        open_ok = self._hold_start[label]["open"] is not None and (now - self._hold_start[label]["open"] >= GESTURE_HOLD_SEC)

        self._fist_buf[label].append(fist_ok)
        self._open_buf[label].append(open_ok)

        cf = self._confirmed(self._fist_buf[label])
        co = self._confirmed(self._open_buf[label])

        # 超时重置
        if (self._phase[label] == self._GOT_FIRST and
                time.time() - self._seq_start[label] > SEQ_WINDOW):
            self._reset(label)

        phase = self._phase[label]

        if phase == self._IDLE:
            if cf:
                self._phase[label]      = self._GOT_FIRST
                self._first_gest[label] = "fist"
                self._seq_start[label]  = time.time()
                self.pending[label]     = "fist"
            elif co:
                self._phase[label]      = self._GOT_FIRST
                self._first_gest[label] = "open"
                self._seq_start[label]  = time.time()
                self.pending[label]     = "open"

        elif phase == self._GOT_FIRST:
            first = self._first_gest[label]
            if first == "fist":
                if co:
                    return "on"   # fist → open = 关
                elif cf:
                    self._seq_start[label] = time.time()
                elif not co and not cf:
                    neither = not fist and not open_hand
                    if neither:
                        self._reset(label)
            else:  # first == "open"
                if cf:
                    return "off"   # open → fist = 开
                elif co:
                    self._seq_start[label] = time.time()
                elif not cf and not co:
                    neither = not fist and not open_hand
                    if neither:
                        self._reset(label)

        return None

    def _reset(self, label):
        self._phase[label]      = self._IDLE
        self._first_gest[label] = None
        self.pending[label]     = None

    def can_trigger(self, label):
        return time.time() - self.last_trigger[label] > GESTURE_COOLDOWN

    def mark(self, label):
        self.last_trigger[label] = time.time()
        self._reset(label)
        self._fist_buf[label].clear()
        self._open_buf[label].clear()


# ═══════════════════════════════════════════════════════
#  精细模式追踪器（按住捏合调节，松开停止）
# ═══════════════════════════════════════════════════════
class FineModeTracker:
    """
    新设计：捏合 = 精细调节激活，松开 = 停止
    - 捏合稳定 PINCH_CONFIRM_SEC 后才激活（防误触）
    - 丢失捏合后有 PINCH_GRACE_SEC 宽限（抗遮挡）
    - 激活时屏蔽同侧粗调
    - 左手 → 亮度，右手 → 音量
    - 灵敏度独立（FINE_SENSITIVITY，比粗调低）
    """
    def __init__(self):
        self.active      = {"Left": False,  "Right": False}
        self._pinch_start= {"Left": None,   "Right": None}  # 捏合开始时间
        self._pinch_lost = {"Left": None,   "Right": None}  # 丢失时间

        self._kf_mid = {
            "Left":  KalmanWristFilter(dt=1/30, process_noise=3e-4, obs_noise=3e-3),
            "Right": KalmanWristFilter(dt=1/30, process_noise=3e-4, obs_noise=3e-3),
        }
        self._buf_y  = {"Left": deque(maxlen=FINE_WIN), "Right": deque(maxlen=FINE_WIN)}
        self._buf_x  = {"Left": deque(maxlen=FINE_WIN), "Right": deque(maxlen=FINE_WIN)}
        self._warmup = {"Left": 0, "Right": 0}   # 激活后预热帧

        self.axis  = {"Left": None, "Right": None}
        self.delta = {"Left": 0.0,  "Right": 0.0}

    @staticmethod
    def _midpoint(hand_lm):
        t = hand_lm.landmark[4]
        i = hand_lm.landmark[8]
        return (t.x + i.x) * 0.5, (t.y + i.y) * 0.5

    def update(self, label, hand_lm):
        """返回 (是否正在精细调节, OK手势进度0~1)"""
        ok = is_ok_gesture(hand_lm)   # OK手势替换捏合

        # ── OK手势状态机 ─────────────────────────
        if ok:
            self._pinch_lost[label] = None
            if self._pinch_start[label] is None:
                self._pinch_start[label] = time.time()
        else:
            if self._pinch_lost[label] is None:
                self._pinch_lost[label] = time.time()
            # 宽限期超时 → 真正松开
            if time.time() - self._pinch_lost[label] > PINCH_GRACE_SEC:
                self._pinch_start[label] = None
                self._pinch_lost[label]  = None
                # 松开 → 退出精细模式，清空缓冲
                if self.active[label]:
                    self.active[label] = False
                    self._buf_y[label].clear()
                    self._buf_x[label].clear()
                    self._kf_mid[label].reset()
                    self._warmup[label] = 0

        ps = self._pinch_start[label]
        progress = 0.0
        if ps is not None:
            elapsed  = time.time() - ps
            progress = clamp(elapsed / PINCH_CONFIRM_SEC, 0.0, 1.0)
            # 稳定确认后进入精细模式
            if elapsed >= PINCH_CONFIRM_SEC and not self.active[label]:
                self.active[label] = True
                self._buf_y[label].clear()
                self._buf_x[label].clear()
                self._kf_mid[label].reset()
                self._warmup[label] = FINE_WIN

        # ── 精细模式控制信号 ─────────────────────
        self.axis[label]  = None
        self.delta[label] = 0.0

        if self.active[label]:
            if self._warmup[label] > 0:
                self._warmup[label] -= 1
            else:
                mx, my = self._midpoint(hand_lm)
                sx, sy = self._kf_mid[label].update(mx, my)
                self._buf_y[label].append(sy)
                self._buf_x[label].append(sx)

                if len(self._buf_y[label]) == FINE_WIN:
                    dy = -(self._buf_y[label][-1] - self._buf_y[label][0])
                    dx =   self._buf_x[label][-1] - self._buf_x[label][0]
                    if abs(dy) >= abs(dx) and abs(dy) > FINE_DEAD_ZONE:
                        self.axis[label]  = "y"
                        self.delta[label] = dy
                    elif abs(dx) > abs(dy) and abs(dx) > FINE_DEAD_ZONE:
                        self.axis[label]  = "x"
                        self.delta[label] = dx

        return self.active[label], progress

# ═══════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════
def put(img, text, x, y, scale=0.5, color=(220,220,220), bold=False):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)

def draw_bg(img, x1, y1, x2, y2, alpha=0.75):
    ov = img.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), (15,15,15), -1)
    cv2.rectangle(ov, (x1,y1), (x2,y2), (55,55,55),  1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

def draw_bar(img, x, y, w, val, vmin, vmax, label, unit, color):
    pct  = clamp((val-vmin)/(vmax-vmin), 0, 1)
    fill = int(w * pct)
    bh   = 20
    cv2.rectangle(img, (x,y), (x+w, y+bh), (30,30,30), -1)
    cv2.rectangle(img, (x,y), (x+fill, y+bh), color,   -1)
    cv2.rectangle(img, (x,y), (x+w, y+bh),   (65,65,65), 1)
    put(img, label, x+5, y+bh-5, 0.38, (190,190,190))
    val_str = f"{int(val)}{unit}"
    sz = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
    put(img, val_str, x+w-sz[0]-5, y+bh-4, 0.48, (255,255,255), bold=True)

def draw_hold_bar(img, x, y, w, progress, label, color):
    fill = int(w * progress)
    bh   = 13
    cv2.rectangle(img, (x,y), (x+w, y+bh), (30,30,30), -1)
    cv2.rectangle(img, (x,y), (x+fill, y+bh), color,   -1)
    cv2.rectangle(img, (x,y), (x+w, y+bh),   (65,65,65), 1)
    put(img, f"{label} {int(progress*100)}%", x+4, y+bh-2, 0.32, (210,210,210))

def draw_zone_arrow(img, cx, cy, zone, depth=0.0):
    """四方向箭头，当前方向高亮，亮度随深度变化"""
    r = 22
    arrows = {
        "up":    [(cx,cy-r),(cx-8,cy-r+13),(cx+8,cy-r+13)],
        "down":  [(cx,cy+r),(cx-8,cy+r-13),(cx+8,cy+r-13)],
        "left":  [(cx-r,cy),(cx-r+13,cy-8),(cx-r+13,cy+8)],
        "right": [(cx+r,cy),(cx+r-13,cy-8),(cx+r-13,cy+8)],
    }
    cv2.line(img, (cx-r,cy), (cx+r,cy), (45,45,45), 1)
    cv2.line(img, (cx,cy-r), (cx,cy+r), (45,45,45), 1)
    for d, pts in arrows.items():
        if d == zone:
            g   = int(140 + 115 * clamp(depth, 0, 1))
            col = (0, g, int(g * 0.65))
        else:
            col = (48, 48, 48)
        cv2.fillPoly(img, [np.array(pts, np.int32)], col)

def draw_panel(frame, state: VirtualState,
               lc: ArmController, rc: ArmController,
               gest: GestureTracker, fine: FineModeTracker,
               hand_visible: dict = None,
               hand_ready: dict = None):
    H, W = frame.shape[:2]
    P  = 10
    PW = 272

    # ── 左面板：台灯 ──────────────────────────
    draw_bg(frame, P, P, P+PW, P+310)

    lc_col = (0,230,100) if state.lamp_on else (95,95,95)
    put(frame, "[ LAMP ]", P+8, P+22, 0.60, lc_col, bold=True)
    put(frame, "ON" if state.lamp_on else "OFF", P+126, P+22, 0.60, lc_col, bold=True)

    draw_bar(frame, P+8, P+32, PW-16,
             state.brightness, B_MIN, B_MAX, "Brightness", "%", (255,200,60))
    draw_bar(frame, P+8, P+60, PW-16,
             state.temp, T_MIN, T_MAX, "Color Temp", "K", (160,200,255))

    # 激活状态
    hv_l    = (hand_visible or {}).get("Left", 0)
    hr_l    = (hand_ready   or {}).get("Left", False)
    hv_ok_l = hv_l >= HAND_MIN_VISIBLE_Y
    act_col = (0,230,100) if (lc.active and hr_l) else ((255,160,0) if lc.active else (70,70,70))
    cv2.circle(frame, (P+16, P+98), 7, act_col, -1)
    put(frame, "CTRL" if lc.active else "idle", P+28, P+102, 0.38, act_col, bold=lc.active)
    hand_str_l = f"hand={hv_l}pts {'▶' if hr_l else '…'}"
    put(frame, hand_str_l, P+90, P+102, 0.33,
        (80,200,80) if hr_l else (200,140,0))
    # 运动轨迹调试信息
    axis_str  = lc.axis if lc.axis else "—"
    delta_str = f"{lc.delta:+.3f}" if lc.axis else "0.000"
    frz_col   = (255,80,50) if lc.frozen else (70,70,70)
    put(frame, f"axis:{axis_str}  delta={delta_str}", P+8, P+120, 0.38,
        (0,200,150) if lc.axis else (65,65,65))
    put(frame, f"weight:{lc.weight:.2f}  dv={lc.calc_delta_value():+.2f}",
        P+8, P+138, 0.36, (140,140,140))
    put(frame, f"{'[FROZEN]' if lc.frozen else 'running'}",
        P+8, P+154, 0.38, frz_col, bold=lc.frozen)
    # 意图分数
    i_score_l = lc.intent_score
    i_col_l   = (0,200,100) if i_score_l >= INTENT_THRESH else (255,80,50)
    put(frame, f"intent={i_score_l:.2f} {'✓' if i_score_l>=INTENT_THRESH else '✗ BLOCKED'}",
        P+8, P+168, 0.36, i_col_l, bold=(i_score_l < INTENT_THRESH))
    put(frame, f"sw={lc.shoulder_width:.3f}  angle={lc.angle:.1f}d",
        P+8, P+182, 0.32, (100,100,100))

    # 序列手势状态
    pending_l = gest.pending.get("Left")
    seq_col   = (255,180,0) if pending_l else (60,60,60)
    put(frame, f"seq: {pending_l} -> ?" if pending_l else "seq: waiting",
        P+8, P+186, 0.38, seq_col)
    lf = gest._confirmed(gest._fist_buf["Left"])
    lo = gest._confirmed(gest._open_buf["Left"])
    put(frame, f"fist:{'OK' if lf else '--'}  open:{'OK' if lo else '--'}",
        P+8, P+202, 0.36, (120,120,120))

    # 精细模式
    l_fine    = fine.active["Left"]
    fine_col  = (0, 200, 255) if l_fine else (60, 60, 60)
    fine_bg   = (0, 40, 60)   if l_fine else (20, 20, 20)
    if l_fine:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (P+5, P+220), (P+PW-5, P+248), fine_bg, -1)
        cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
    put(frame, "FINE" if l_fine else "fine: off", P+8, P+234, 0.42, fine_col, bold=l_fine)
    if l_fine:
        fa = fine.axis["Left"] or "—"
        fd = f"{fine.delta['Left']:+.3f}" if fine.axis["Left"] else "0.000"
        put(frame, f"axis:{fa}  delta={fd}", P+65, P+234, 0.36, (0,200,255))
    put(frame, "OK gesture = fine mode", P+8, P+250, 0.29, (75,75,75))

    put(frame, f"Activate : amp_y>{ACTIVATE_THRESH} (chest ref)",  P+8, P+264, 0.30, (80,80,80))
    put(frame, f"Deactivate: amp_y<{DEACTIVATE_THRESH}",           P+8, P+277, 0.30, (80,80,80))
    put(frame, f"motion: k={MOTION_K} sens={MOTION_SENSITIVITY}",  P+8, P+290, 0.30, (80,80,80))
    put(frame, "y-axis=Bright  x-axis=Temp  frozen=lock",          P+8, P+303, 0.29, (70,70,70))

    # ── 右面板：音箱 ──────────────────────────
    rx = W - P - PW
    draw_bg(frame, rx, P, rx+PW, P+310)

    sc_col = (100,180,255) if state.speaker_playing else (95,95,95)
    put(frame, "[ SPEAKER ]", rx+8, P+22, 0.60, sc_col, bold=True)
    put(frame, "PLAY" if state.speaker_playing else "STOP",
        rx+160, P+22, 0.60, sc_col, bold=True)

    draw_bar(frame, rx+8, P+32, PW-16,
             state.volume, 0, 100, "Volume", "%", (100,200,255))
    put(frame, f"Track: #{state.track}", rx+8, P+68, 0.46, (200,200,200))

    hv_r    = (hand_visible or {}).get("Right", 0)
    hr_r    = (hand_ready   or {}).get("Right", False)
    hv_ok_r = hv_r >= HAND_MIN_VISIBLE_Y
    act_col2 = (0,230,100) if (rc.active and hr_r) else ((255,160,0) if rc.active else (70,70,70))
    cv2.circle(frame, (rx+16, P+98), 7, act_col2, -1)
    put(frame, "CTRL" if rc.active else "idle", rx+28, P+102, 0.38, act_col2, bold=rc.active)
    hand_str_r = f"hand={hv_r}pts {'▶' if hr_r else '…'}"
    put(frame, hand_str_r, rx+90, P+102, 0.33,
        (80,200,80) if hr_r else (200,140,0))

    _axis_to_dir = {"y": "up", "x": "right", None: "neutral"}
    draw_zone_arrow(frame, rx+PW-32, P+130,
                    _axis_to_dir.get(rc.axis, "neutral"), rc.weight)
    axis_str2  = rc.axis if rc.axis else "—"
    delta_str2 = f"{rc.delta:+.3f}" if rc.axis else "0.000"
    frz_col2   = (255,80,50) if rc.frozen else (70,70,70)
    put(frame, f"axis:{axis_str2}  delta={delta_str2}", rx+8, P+120, 0.38,
        (0,200,150) if rc.axis else (65,65,65))
    put(frame, f"weight:{rc.weight:.2f}  dv={rc.calc_delta_value():+.2f}",
        rx+8, P+138, 0.36, (140,140,140))
    put(frame, f"{'[FROZEN]' if rc.frozen else 'running'}",
        rx+8, P+154, 0.38, frz_col2, bold=rc.frozen)
    i_score_r = rc.intent_score
    i_col_r   = (0,200,100) if i_score_r >= INTENT_THRESH else (255,80,50)
    put(frame, f"intent={i_score_r:.2f} {'✓' if i_score_r>=INTENT_THRESH else '✗ BLOCKED'}",
        rx+8, P+168, 0.36, i_col_r, bold=(i_score_r < INTENT_THRESH))
    put(frame, f"sw={rc.shoulder_width:.3f}  angle={rc.angle:.1f}d",
        rx+8, P+182, 0.32, (100,100,100))

    # 序列手势状态
    pending_r = gest.pending.get("Right")
    seq_col2  = (255,180,0) if pending_r else (60,60,60)
    put(frame, f"seq: {pending_r} -> ?" if pending_r else "seq: waiting",
        rx+8, P+186, 0.38, seq_col2)
    rf = gest._confirmed(gest._fist_buf["Right"])
    ro = gest._confirmed(gest._open_buf["Right"])
    put(frame, f"fist:{'OK' if rf else '--'}  open:{'OK' if ro else '--'}",
        rx+8, P+202, 0.36, (120,120,120))

    # 精细模式
    r_fine    = fine.active["Right"]
    fine_col2 = (0, 200, 255) if r_fine else (60, 60, 60)
    fine_bg2  = (0, 40, 60)   if r_fine else (20, 20, 20)
    if r_fine:
        ov3 = frame.copy()
        cv2.rectangle(ov3, (rx+5, P+220), (rx+PW-5, P+248), fine_bg2, -1)
        cv2.addWeighted(ov3, 0.6, frame, 0.4, 0, frame)
    put(frame, "FINE" if r_fine else "fine: off", rx+8, P+234, 0.42, fine_col2, bold=r_fine)
    if r_fine:
        fa2 = fine.axis["Right"] or "—"
        fd2 = f"{fine.delta['Right']:+.3f}" if fine.axis["Right"] else "0.000"
        put(frame, f"axis:{fa2}  delta={fd2}", rx+65, P+234, 0.36, (0,200,255))
    put(frame, "OK gesture = fine mode", rx+8, P+250, 0.29, (75,75,75))

    put(frame, f"Activate : amp_y>{ACTIVATE_THRESH} (chest ref)", rx+8, P+264, 0.30, (80,80,80))
    put(frame, f"Deactivate: amp_y<{DEACTIVATE_THRESH}",          rx+8, P+277, 0.30, (80,80,80))
    put(frame, f"motion: k={MOTION_K} sens={MOTION_SENSITIVITY}", rx+8, P+290, 0.30, (80,80,80))
    put(frame, "y-axis=Volume  x-axis=Track  frozen=lock",        rx+8, P+303, 0.29, (70,70,70))

    # ── 底部日志 ──────────────────────────────
    log_rows = max(len(state.log), 1)
    log_h    = log_rows * 18 + 28
    log_y0   = H - P - log_h
    draw_bg(frame, P, log_y0, P+420, H-P)
    put(frame, "ACTION LOG", P+8, log_y0+15, 0.38, (120,120,120))
    for i, entry in enumerate(state.log):
        fade = int(220*(1-i/log_rows*0.6))
        col  = (0,fade,int(fade*0.8)) if i==0 else (fade,fade,fade)
        put(frame, entry, P+8, log_y0+30+i*18, 0.36, col)

    # ── 中央 Flash ────────────────────────────
    elapsed = time.time() - state.last_action_ts
    if elapsed < 1.0 and state.last_action:
        a   = max(0.0, 1.0 - elapsed)
        col = (0, int(255*a), int(200*a))
        sz  = cv2.getTextSize(state.last_action, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        tx, ty = (W-sz[0])//2, H//2
        cv2.putText(frame, state.last_action,(tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame, state.last_action,(tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.85,col,2,cv2.LINE_AA)

# ═══════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════
GESTURE_TASK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task"
)

def run(worker: MijiaWorker):
    _load_intent_model()   # 加载意图模型（文件不存在时静默跳过）
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_sty   = mp.solutions.drawing_styles

    pose  = mp_pose.Pose(min_detection_confidence=0.6,
                         min_tracking_confidence=0.6,
                         model_complexity=0)          # 0=轻量模型，帧率更高
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)

    # ── GestureRecognizer（用于 fist / open 判定）──────────
    _gesture_recognizer = None
    if os.path.exists(GESTURE_TASK_PATH):
        with open(GESTURE_TASK_PATH, "rb") as f:
            task_bytes = f.read()
        _gr_opts = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=task_bytes),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(_gr_opts)
        print(f"✅ GestureRecognizer 已加载(buffer): {GESTURE_TASK_PATH}")
    else:
        print(f"⚠️  未找到 {GESTURE_TASK_PATH}，fist/open 回退到几何规则")
    # 每帧刷新：{"Left": "Closed_Fist"/"Open_Palm"/None, "Right": ...}
    _gr_result = {"Left": None, "Right": None}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    lc    = ArmController("left")
    rc    = ArmController("right")
    gest  = GestureTracker()
    fine  = FineModeTracker()
    state = VirtualState()

    prev_raxis    = None
    last_log_b    = last_log_v = 0.0
    LOG_INTERVAL  = 0.4
    palm_pts      = {"Left": None, "Right": None}

    # 单次激活基准值（激活瞬间记录，用于计算本次累计变化量）
    session_base_b = None   # 本次激活时的亮度初始值
    session_base_v = None   # 本次激活时的音量初始值
    session_base_t = None   # 本次激活时的色温初始值
    lc_was_active  = False
    rc_was_active  = False

    # 连续检测帧计数器
    hand_stable  = {"Left": 0, "Right": 0}   # 连续有效检测帧数
    hand_lost_ct = {"Left": 0, "Right": 0}   # 连续丢失帧数
    hand_ready   = {"Left": False, "Right": False}  # 是否达到稳定
    total_frames = 0

    PL  = mp_pose.PoseLandmark
    IDX = {
        "LS": PL.LEFT_SHOULDER,  "LE": PL.LEFT_ELBOW,  "LW": PL.LEFT_WRIST,
        "RS": PL.RIGHT_SHOULDER, "RE": PL.RIGHT_ELBOW, "RW": PL.RIGHT_WRIST,
        "LH": PL.LEFT_HIP,       "RH": PL.RIGHT_HIP,
    }

    print("═"*55)
    print("  运动轨迹版：激活区 + 位移积分 + 抛物线速率")
    print("═"*55)
    print(f"  激活：amp_y > {ACTIVATE_THRESH}（手腕高于胸部代理点）")
    print(f"  退出：amp_y < {DEACTIVATE_THRESH}（手臂放下）")
    print(f"  控制：5帧位移积分，胸口速率最高（k={MOTION_K}）")
    print(f"  放手保护：快速下落冻结(D) + 连续下落冻结(B)")
    print(f"  手势：open->fist=开，fist->open=关（2秒窗口）")
    print("  按 Q 退出")
    print("═"*55)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]

        # 全分辨率推理：远距离手势细节更好，但速度会下降
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        pose_res  = pose.process(rgb_frame)
        hands_res = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        now = time.time()

        # ── GestureRecognizer 推理（每帧）──────────────────
        _gr_result = {"Left": None, "Right": None}
        if _gesture_recognizer is not None:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            gr_out = _gesture_recognizer.recognize(mp_img)
            # gr_out.gestures[i][0].category_name 对应第 i 只手
            # gr_out.handedness[i][0].display_name 为 "Left"/"Right"
            for g_list, h_list in zip(gr_out.gestures, gr_out.handedness):
                side     = h_list[0].display_name          # "Left" / "Right"
                cat_name = g_list[0].category_name         # e.g. "Closed_Fist"
                _gr_result[side] = cat_name

        # ── 每帧收集手掌五点平均坐标（0,5,9,13,17），供 ArmController 使用 ──
        # 五点覆盖手掌底边，对手指弯曲不敏感，任何单点跳动都被平均掉
        PALM_BASE  = [0, 5, 9, 13, 17]
        palm_pts   = {"Left": None,  "Right": None}
        hand_visible = {"Left": 0,   "Right": 0}   # 该侧检测到的关键点数
        if hands_res.multi_hand_landmarks:
            for hlm, hinfo in zip(hands_res.multi_hand_landmarks,
                                  hands_res.multi_handedness):
                label   = hinfo.classification[0].label
                palm_x  = float(np.mean([hlm.landmark[i].x for i in PALM_BASE]))
                palm_y  = float(np.mean([hlm.landmark[i].y for i in PALM_BASE]))
                palm_pts[label]    = np.array([palm_x, palm_y])
                # 统计可靠关键点数：优先用 presence score，旧版 MediaPipe 回退到坐标边界
                hand_visible[label] = sum(
                    1 for lmk in hlm.landmark
                    if getattr(lmk, 'presence', None) is not None
                    and lmk.presence > 0.5
                ) or sum(
                    1 for lmk in hlm.landmark
                    if 0.0 < lmk.x < 1.0 and 0.0 < lmk.y < 1.0
                )

        # ── 连续检测帧计数更新 ────────────────────────
        for side in ["Left", "Right"]:
            if hand_visible[side] >= HAND_MIN_VISIBLE_Y:
                hand_lost_ct[side] = 0
                hand_stable[side]  = min(hand_stable[side] + 1, HAND_STABLE_FRAMES * 3)
            else:
                hand_lost_ct[side] += 1
                if hand_lost_ct[side] > HAND_LOST_GRACE:
                    hand_stable[side] = 0
            hand_ready[side] = hand_stable[side] >= HAND_STABLE_FRAMES

        # ── Pose ──────────────────────────────────
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            def gn(k): return np.array([lm[IDX[k]].x, lm[IDX[k]].y])

            ls, le, lw = gn("LS"), gn("LE"), gn("LW")
            rs, re, rw = gn("RS"), gn("RE"), gn("RW")
            lh, rh     = gn("LH"), gn("RH")

            lc.update(ls, le, lw, rs, lh, palm_pt=palm_pts["Left"])
            rc.update(rs, re, rw, ls, rh, palm_pt=palm_pts["Right"])

            dt_l = lc.get_dt()
            dt_r = rc.get_dt()

            # ── 单次激活基准值记录 ────────────────────
            if lc.active and not lc_was_active:
                session_base_b = state.brightness
                session_base_t = state.temp
            if rc.active and not rc_was_active:
                session_base_v = state.volume
            if not lc.active:
                session_base_b = None
                session_base_t = None
            if not rc.active:
                session_base_v = None
            lc_was_active = lc.active
            rc_was_active = rc.active

            # ── 左臂：亮度（y轴）+ 色温（x轴）
            # 精细模式激活时跳过粗调，避免撞车
            if lc.active and not fine.active.get("Left", False) and hand_ready["Left"]:
                dv_l = lc.calc_delta_value()
                hv_l = hand_visible["Left"]
                if lc.axis == "y" and dv_l != 0.0 and hv_l >= HAND_MIN_VISIBLE_Y:
                    raw_b  = state.brightness + dv_l
                    # 速率限制：每秒最多变化 MAX_RATE_B %
                    new_b  = ArmController.rate_limit(state.brightness, raw_b, MAX_RATE_B, dt_l)
                    new_b  = clamp(new_b, B_MIN, B_MAX)
                    new_b  = lc.dead_zone(state.brightness, new_b, DEAD_ZONE_B)
                    if session_base_b is not None:
                        if new_b > session_base_b + SESSION_CAP_B and dv_l > 0: new_b = session_base_b + SESSION_CAP_B
                        if new_b < session_base_b - SESSION_CAP_B and dv_l < 0: new_b = session_base_b - SESSION_CAP_B
                    state.brightness = new_b
                    worker.send("brightness", state.brightness)
                    if now - last_log_b > LOG_INTERVAL:
                        state.push_log(f"Brightness = {int(state.brightness)}%")
                        last_log_b = now
                elif lc.axis == "x" and dv_l != 0.0 and hv_l >= HAND_MIN_VISIBLE_X:
                    # dv_l 已经是 x 轴的变化量（用 MOTION_SENSITIVITY_X=30 算的）
                    # 色温范围 2700-6500K（3800K），按比例放大
                    raw_t  = state.temp - dv_l * 40.0  # 方向反转 + 幅度再加大：横向位移更小也能明显改变色温
                    new_t  = ArmController.rate_limit(state.temp, raw_t, MAX_RATE_T, dt_l)
                    new_t  = clamp(new_t, T_MIN, T_MAX)
                    new_t  = lc.dead_zone(state.temp, new_t, DEAD_ZONE_T)
                    if session_base_t is not None:
                        if new_t > session_base_t + SESSION_CAP_T and dv_l > 0: new_t = session_base_t + SESSION_CAP_T
                        if new_t < session_base_t - SESSION_CAP_T and dv_l < 0: new_t = session_base_t - SESSION_CAP_T
                    state.temp = new_t
                    worker.send("temp", state.temp)
                    if now - last_log_b > LOG_INTERVAL:
                        state.push_log(f"Temp = {int(state.temp)}K")
                        last_log_b = now

            # ── 右臂：音量（y轴）+ 换曲（x轴）
            if rc.active and not fine.active.get("Right", False) and hand_ready["Right"]:
                dv_r = rc.calc_delta_value()
                hv_r = hand_visible["Right"]
                if rc.axis == "y" and dv_r != 0.0 and hv_r >= HAND_MIN_VISIBLE_Y:
                    raw_v  = state.volume + dv_r
                    new_v  = ArmController.rate_limit(state.volume, raw_v, MAX_RATE_V, dt_r)
                    new_v  = clamp(new_v, 0, 100)
                    new_v  = rc.dead_zone(state.volume, new_v, DEAD_ZONE_V)
                    if session_base_v is not None:
                        if new_v > session_base_v + SESSION_CAP_V and dv_r > 0: new_v = session_base_v + SESSION_CAP_V
                        if new_v < session_base_v - SESSION_CAP_V and dv_r < 0: new_v = session_base_v - SESSION_CAP_V
                    state.volume = new_v
                    worker.send("volume", state.volume)
                    if now - last_log_v > LOG_INTERVAL:
                        state.push_log(f"Volume = {round(state.volume, 1)}%")
                        last_log_v = now
                elif rc.axis == "x" and dv_r != 0.0:
                    if prev_raxis != "x":
                        if rc.delta > 0:
                            state.track += 1
                            state.push_log(f"Next Track -> #{state.track}")
                            worker.send("next_track", 1)
                        else:
                            state.track = max(1, state.track - 1)
                            state.push_log(f"Prev Track -> #{state.track}")
                            worker.send("prev_track", 1)

            prev_raxis = rc.axis

            # ── 骨架绘制 ──────────────────────────
            mp_draw.draw_landmarks(
                frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(0,200,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0,150,220), thickness=2))

            # 肩肘腕三角形
            for ctrl, pts, col in [
                (lc, [ls,le,lw], (80,255,120)),
                (rc, [rs,re,rw], (255,160,50))
            ]:
                poly  = np.array([[int(p[0]*W), int(p[1]*H)] for p in pts])
                thick = 3 if ctrl.active else 1
                cv2.polylines(frame, [poly], True, col, thick)
                ang = calc_angle(pts[0], pts[1], pts[2])
                mx, my = int(pts[1][0]*W), int(pts[1][1]*H)
                put(frame, f"{ang:.0f}d", mx+6, my-6, 0.33, col)

            # 肩到髋参考线
            for s, h, col in [(ls,lh,(80,255,120)),(rs,rh,(255,160,50))]:
                cv2.line(frame, (int(s[0]*W),int(s[1]*H)),
                                (int(h[0]*W),int(h[1]*H)), col, 1)


            # 手腕标注
            for ctrl, wpt, col in [(lc,lw,(80,255,120)),(rc,rw,(255,160,50))]:
                wx2, wy2 = int(wpt[0]*W), int(wpt[1]*H)
                if ctrl.active:
                    lbl = ctrl.axis if ctrl.axis else "hold"
                    zc  = (0,255,180) if ctrl.axis else (160,160,160)
                    frz = " [FRZ]" if ctrl.frozen else ""
                    put(frame, lbl + frz, wx2+8, wy2-8, 0.40, zc, bold=bool(ctrl.axis))
                else:
                    put(frame, "idle", wx2+8, wy2-8, 0.33, (90,90,90))

        # ── Hands ──────────────────────────────────
        if hands_res.multi_hand_landmarks:
            for hlm, hinfo in zip(hands_res.multi_hand_landmarks,
                                  hands_res.multi_handedness):
                label     = hinfo.classification[0].label
                _gr_cat = _gr_result.get(label) if _gesture_recognizer is not None else None

                # 使用 standalone 验证通过的 no-facing 几何规则作为主判定
                open_hand, fist, _gf_dbg = classify_open_fist_nofacing(hlm)

                # 精细模式更新
                f_active, f_prog = fine.update(label, hlm)

                # 精细模式控制输出：加 hand_ready 门控，且同一时刻只允许一侧输出
                other = "Right" if label == "Left" else "Left"
                fine_blocked = (not hand_ready[label]) or fine.active.get(other, False)

                if f_active and not fine_blocked:
                    f_axis  = fine.axis[label]
                    f_delta = fine.delta[label]
                    if label == "Left":
                        if f_axis == "y" and f_delta != 0.0:
                            new_b = clamp(state.brightness + f_delta * FINE_SENSITIVITY,
                                          B_MIN, B_MAX)
                            if session_base_b is not None:
                                if new_b > session_base_b + SESSION_CAP_B and f_delta > 0:
                                    new_b = session_base_b + SESSION_CAP_B
                                if new_b < session_base_b - SESSION_CAP_B and f_delta < 0:
                                    new_b = session_base_b - SESSION_CAP_B
                            state.brightness = new_b
                            worker.send("brightness", state.brightness)
                            if now - last_log_b > LOG_INTERVAL:
                                state.push_log(f"[FINE] Brightness = {int(state.brightness)}%")
                                last_log_b = now
                        elif f_axis == "x" and f_delta != 0.0:
                            raw_t  = state.temp + f_delta * FINE_SENSITIVITY * 8.0
                            new_t  = clamp(raw_t, T_MIN, T_MAX)
                            new_t  = lc.dead_zone(state.temp, new_t, DEAD_ZONE_T)
                            if session_base_t is not None:
                                if new_t > session_base_t + SESSION_CAP_T and f_delta > 0:
                                    new_t = session_base_t + SESSION_CAP_T
                                if new_t < session_base_t - SESSION_CAP_T and f_delta < 0:
                                    new_t = session_base_t - SESSION_CAP_T
                            state.temp = new_t
                            worker.send("temp", state.temp)
                            if now - last_log_b > LOG_INTERVAL:
                                state.push_log(f"[FINE] Temp = {int(state.temp)}K")
                                last_log_b = now
                    else:
                        if f_axis == "y" and f_delta != 0.0:
                            new_v = clamp(state.volume + f_delta * FINE_SENSITIVITY,
                                          0, 100)
                            if session_base_v is not None:
                                if new_v > session_base_v + SESSION_CAP_V and f_delta > 0:
                                    new_v = session_base_v + SESSION_CAP_V
                                if new_v < session_base_v - SESSION_CAP_V and f_delta < 0:
                                    new_v = session_base_v - SESSION_CAP_V
                            state.volume = new_v
                            worker.send("volume", state.volume)
                            if now - last_log_v > LOG_INTERVAL:
                                state.push_log(f"[FINE] Volume = {int(state.volume)}%")
                                last_log_v = now

                # OK手势进度条
                if f_prog > 0.05 and not f_active:
                    wx_p = int(hlm.landmark[8].x * W)
                    wy_p = int(hlm.landmark[8].y * H)
                    bar_w = 60
                    fill  = int(bar_w * f_prog)
                    cv2.rectangle(frame, (wx_p-bar_w//2, wy_p-20),
                                  (wx_p+bar_w//2, wy_p-10), (30,30,30), -1)
                    cv2.rectangle(frame, (wx_p-bar_w//2, wy_p-20),
                                  (wx_p-bar_w//2+fill, wy_p-10), (0,200,255), -1)
                    put(frame, "OK", wx_p-bar_w//2, wy_p-22, 0.30, (0,200,255))

                # 序列手势（精细模式激活时跳过）
                if not f_active:
                    result = gest.update(label, fist, open_hand)
                    if result and gest.can_trigger(label):
                        if result == "on":
                            if label == "Left":
                                state.lamp_on = True
                                state.push_log("Left open->fist -> Lamp ON")
                                worker.send("lamp_on", True)
                            else:
                                state.speaker_playing = True
                                state.push_log("Right open->fist -> Speaker PLAY")
                                worker.send("play", 1)
                        elif result == "off":
                            if label == "Left":
                                state.lamp_on = False
                                state.push_log("Left fist->open -> Lamp OFF")
                                worker.send("lamp_on", False)
                            else:
                                state.speaker_playing = False
                                state.push_log("Right fist->open -> Speaker STOP")
                                worker.send("stop", 1)
                        gest.mark(label)

                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_sty.get_default_hand_landmarks_style(),
                    mp_sty.get_default_hand_connections_style())

                wx3 = int(hlm.landmark[0].x * W)
                wy3 = int(hlm.landmark[0].y * H)
                if f_active:
                    g_lbl = "FINE"
                    col   = (0, 200, 255)
                else:
                    g_lbl = "fist" if fist else ("open" if open_hand else "-")
                    if _gr_cat in ("Closed_Fist", "Open_Palm"):
                        g_lbl += f"|GR:{'fist' if _gr_cat=='Closed_Fist' else 'open'}"
                    col   = (0,230,120) if label=="Left" else (255,150,50)
                put(frame, f"{label}[{g_lbl}]", wx3-30, wy3+28, 0.44, col, bold=f_active)

        draw_panel(frame, state, lc, rc, gest, fine,
                   hand_visible=hand_visible, hand_ready=hand_ready)

        cv2.imshow("Mijia Gesture Controller [DEMO]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        total_frames += 1

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    if _gesture_recognizer is not None:
        _gesture_recognizer.close()

def get_api():
    if not os.path.exists("jsons"):
        os.mkdir("jsons")
    auth_path = "jsons/auth.json"
    if os.path.exists(auth_path):
        with open(auth_path) as f:
            auth = json.load(f)
        print("✅ 已加载现有登录信息")
    else:
        print("🔐 未找到 auth.json，正在扫码登录...")
        auth = mijiaLogin().QRlogin()
        with open(auth_path, "w") as f:
            json.dump(auth, f, indent=2)
        print("✅ 登录成功，已保存到 auth.json")
    return mijiaAPI(auth)


def select_device(api, prompt):
    devices = api.get_devices_list()
    print(f"\n📱 {prompt}")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d['name']}  ({d['model']})")
    idx = int(input("请输入设备编号: "))
    return devices[idx]


if __name__ == "__main__":
    print("🚀 启动米家手势控制器")
    api = get_api()

    devices = api.get_devices_list()
    print("\n📱 检测到以下设备：")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d['name']}  ({d['model']})")

    lamp_idx = int(input("\n请输入台灯设备编号(例如 0 或 1): ").strip())
    speaker_idx = int(input("请输入音箱设备编号(例如 0 或 1): ").strip())

    lamp_info    = devices[lamp_idx]
    speaker_info = devices[speaker_idx]
    print(f"\n✅ 台灯：[{lamp_idx}] {lamp_info['name']}")
    print(f"✅ 音箱：[{speaker_idx}] {speaker_info['name']}")

    lamp_dev     = mijiaDevice(api, dev_name=lamp_info["name"],    sleep_time=0.1)
    speaker_dev  = mijiaDevice(api, dev_name=speaker_info["name"], sleep_time=0.1)

    worker = MijiaWorker(lamp_dev, speaker_dev)
    print("📷 启动摄像头...\n")
    run(worker)
