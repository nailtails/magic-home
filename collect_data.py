"""
手势意图数据采集脚本
=====================
采集特征：amp_y, vel_y, delta_y, delta_x, weight, active, frozen
标签：1 = 有意调节，0 = 无意动作（放手/抖动/误触）

按键说明（全局，不需要点击窗口）：
  R       — 开始录制当前段
  1       — 停止并标记为【有意调节】
  0       — 停止并标记为【无意动作】
  ESC     — 丢弃当前段（录坏了重来）
  Q       — 退出，保存所有数据

数据保存到 gesture_data.csv，每次运行追加，不覆盖。
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import keyboard
from collections import deque
from filterpy.kalman import KalmanFilter

# ── 与主程序保持一致的参数 ──────────────────────────
ACTIVATE_THRESH   = 0.15
DEACTIVATE_THRESH = 0.02
MOTION_WIN        = 5
MOTION_DEAD_ZONE  = 0.018
MOTION_K          = 4.0
MOTION_MIN_RATE   = 0.10
MOTION_PEAK       = 0.25
DROP_VEL_THRESH   = 0.022
DROP_STREAK       = 4
PALM_BASE         = [0, 5, 9, 13, 17]

WINDOW_SIZE       = 20    # 每个样本的帧数（约0.67秒@30fps）
OUTPUT_CSV        = "gesture_data.csv"
CSV_HEADER        = [
    "frame",
    "amp_y", "vel_y", "delta_y", "delta_x",
    "weight", "active", "frozen",
    "label",   # 1=有意, 0=无意, -1=未标注
    "segment"  # 段编号，同一段连续帧共享同一个label
]

# ── 工具 ─────────────────────────────────────────────
def clamp(v, lo, hi): return max(lo, min(hi, v))

class KalmanFilter2D:
    def __init__(self, dt=1/30, pn=8e-4, on=8e-3):
        from filterpy.kalman import KalmanFilter as KF
        self.kf = KF(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.kf.P = np.eye(4) * 0.1
        self.kf.R = np.eye(2) * on
        self.kf.Q = np.eye(4) * pn
        self._init = False

    def update(self, x, y):
        if not self._init:
            self.kf.x = np.array([x, y, 0., 0.])
            self._init = True
            return x, y
        self.kf.predict()
        self.kf.update(np.array([x, y]))
        return float(self.kf.x[0]), float(self.kf.x[1])

    @property
    def velocity(self):
        return float(self.kf.x[2]), float(self.kf.x[3])

    def reset(self):
        self._init = False
        self.kf.P = np.eye(4) * 0.1

class ArmState:
    """轻量版ArmController，只提取特征，不控制设备"""
    def __init__(self, side):
        self.side   = side
        self.active = False
        self.frozen = False
        self.kf     = KalmanFilter2D()
        self.amp_y  = 0.0
        self.vel_y  = 0.0
        self._buf_y = deque(maxlen=MOTION_WIN)
        self._buf_x = deque(maxlen=MOTION_WIN)
        self.delta_y = 0.0
        self.delta_x = 0.0
        self.weight  = 0.0
        self._drop_streak    = 0
        self._deact_countdown = 0
        self.shoulder_width  = 0.18

    def update(self, shoulder, elbow, wrist, opp, hip, palm_pt=None):
        sw = max(abs(shoulder[0] - opp[0]), 0.05)
        self.shoulder_width = sw
        chest = 0.65 * shoulder + 0.35 * hip

        ctrl_x = palm_pt[0] if palm_pt is not None else (wrist[0]+elbow[0])/2
        ctrl_y = palm_pt[1] if palm_pt is not None else (wrist[1]+elbow[1])/2

        raw_y = (chest[1] - ctrl_y) / sw
        raw_x_b = (ctrl_x - shoulder[0]) / sw
        raw_x = raw_x_b if self.side == "right" else -raw_x_b

        just_activated = False
        if not self.active:
            self.amp_y = raw_y; self.vel_y = 0.0
            if raw_y > ACTIVATE_THRESH:
                self.active = True; just_activated = True
                self.kf.reset(); self.kf.update(ctrl_x, ctrl_y)
                self.frozen = False; self._drop_streak = 0
        else:
            if raw_y < DEACTIVATE_THRESH:
                self._deact_countdown += 1
                self.frozen = True
                if self._deact_countdown >= 5:
                    self.active = False; self.frozen = False
                    self._drop_streak = 0; self._deact_countdown = 0
                    self._buf_y.clear(); self._buf_x.clear()
                self.amp_y = raw_y
                return
            else:
                self._deact_countdown = 0

            fx, fy = self.kf.update(ctrl_x, ctrl_y)
            _, kvy = self.kf.velocity
            self.amp_y = (chest[1] - fy) / sw
            self.vel_y = -kvy / sw

        if self.active:
            self._buf_y.append(self.amp_y)
            self._buf_x.append(raw_x)

            if len(self._buf_y) >= MOTION_WIN:
                self.delta_y = self._buf_y[-1] - self._buf_y[0]
                self.delta_x = self._buf_x[-1] - self._buf_x[0]
            else:
                self.delta_y = 0.0; self.delta_x = 0.0

            raw_w = 1.0 - MOTION_K * ((self.amp_y - MOTION_PEAK) ** 2)
            self.weight = max(MOTION_MIN_RATE, min(1.0, raw_w))

            if not just_activated:
                if self.vel_y < -DROP_VEL_THRESH:
                    self.frozen = True
                if self.vel_y < -MOTION_DEAD_ZONE:
                    self._drop_streak += 1
                else:
                    self._drop_streak = 0
                if self._drop_streak >= DROP_STREAK:
                    self.frozen = True
                if self.vel_y > MOTION_DEAD_ZONE * 1.5:
                    self.frozen = False; self._drop_streak = 0

    def features(self):
        return {
            "amp_y":   round(self.amp_y,  4),
            "vel_y":   round(self.vel_y,  4),
            "delta_y": round(self.delta_y, 4),
            "delta_x": round(self.delta_x, 4),
            "weight":  round(self.weight,  4),
            "active":  int(self.active),
            "frozen":  int(self.frozen),
        }


def put(img, text, x, y, scale=0.5, color=(220,220,220), bold=False):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


def main():
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    pose  = mp_pose.Pose(model_complexity=0,
                         min_detection_confidence=0.6,
                         min_tracking_confidence=0.6)
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)   # 不用 CAP_DSHOW，兼容性更好
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 强制创建窗口，不等第一帧
    cv2.namedWindow("Gesture Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Data Collector", 960, 540)

    larm = ArmState("left")
    rarm = ArmState("right")

    PL  = mp_pose.PoseLandmark
    IDX = {
        "LS": PL.LEFT_SHOULDER,  "LE": PL.LEFT_ELBOW,  "LW": PL.LEFT_WRIST,
        "RS": PL.RIGHT_SHOULDER, "RE": PL.RIGHT_ELBOW, "RW": PL.RIGHT_WRIST,
        "LH": PL.LEFT_HIP,       "RH": PL.RIGHT_HIP,
    }

    # ── CSV初始化 ───────────────────────────────────────
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file    = open(OUTPUT_CSV, "a", newline="")
    writer      = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if not file_exists:
        writer.writeheader()

    # ── 录制状态 ────────────────────────────────────────
    recording    = False
    current_seg  = []       # 当前段的帧列表
    segment_id   = 0
    total_frames = 0
    stats        = {1: 0, 0: 0}   # 已录制的各类帧数
    _key_held    = {}              # 防止长按重复触发

    print("="*50)
    print("  手势意图数据采集")
    print("  R       — 开始录制")
    print("  1       — 停止 + 标记【有意调节】")
    print("  0       — 停止 + 标记【无意动作】")
    print("  ESC     — 丢弃当前段")
    print("  Q       — 退出保存")
    print("="*50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]

        small     = cv2.resize(frame, (W//2, H//2))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb_small.flags.writeable = False
        pose_res  = pose.process(rgb_small)
        hands_res = hands.process(rgb_small)
        rgb_small.flags.writeable = True

        # ── 收集手掌五点均值 ────────────────────────────
        palm_pts = {"Left": None, "Right": None}
        if hands_res.multi_hand_landmarks:
            for hlm, hinfo in zip(hands_res.multi_hand_landmarks,
                                  hands_res.multi_handedness):
                label  = hinfo.classification[0].label
                palm_x = float(np.mean([hlm.landmark[i].x for i in PALM_BASE]))
                palm_y = float(np.mean([hlm.landmark[i].y for i in PALM_BASE]))
                palm_pts[label] = np.array([palm_x, palm_y])

        # ── Pose 更新 ───────────────────────────────────
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            def gn(k): return np.array([lm[IDX[k]].x, lm[IDX[k]].y])
            ls,le,lw = gn("LS"),gn("LE"),gn("LW")
            rs,re,rw = gn("RS"),gn("RE"),gn("RW")
            lh,rh    = gn("LH"),gn("RH")

            larm.update(ls,le,lw,rs,lh, palm_pt=palm_pts["Left"])
            rarm.update(rs,re,rw,ls,rh, palm_pt=palm_pts["Right"])

            mp_draw.draw_landmarks(
                frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,200,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,150,220), thickness=2))

        # ── 录制当前帧特征 ──────────────────────────────
        if recording:
            for arm in [larm, rarm]:
                feat = arm.features()
                feat.update({
                    "frame":   total_frames,
                    "label":   -1,
                    "segment": segment_id,
                })
                current_seg.append(feat)
            total_frames += 1

        # ── UI ─────────────────────────────────────────
        # 状态栏背景
        cv2.rectangle(frame, (0,0), (W, 90), (15,15,15), -1)
        cv2.rectangle(frame, (0,0), (W, 90), (55,55,55),  1)

        if recording:
            rec_col = (0, 60, 220) if (int(time.time()*2) % 2 == 0) else (0,40,180)
            cv2.circle(frame, (28, 28), 10, rec_col, -1)
            put(frame, "REC", 44, 34, 0.55, (80,160,255), bold=True)
            put(frame, f"{len(current_seg)//2} frames", 100, 34, 0.45, (160,160,160))
            put(frame, "press  1 = intentional    0 = unintentional    ESC = discard",
                44, 62, 0.40, (180,180,180))
        else:
            put(frame, "IDLE", 20, 34, 0.55, (120,120,120))
            put(frame, "press  R = start recording    Q = quit & save",
                80, 34, 0.40, (160,160,160))
            put(frame, f"saved:  intentional={stats[1]}f   unintentional={stats[0]}f   "
                       f"total segments={segment_id}",
                20, 62, 0.38, (100,200,100))

        # 手臂激活状态指示
        for arm, label_str, cx in [(larm,"L",W//2-60),(rarm,"R",W//2+20)]:
            col = (0,230,100) if arm.active else (80,80,80)
            if arm.frozen: col = (0,100,255)
            put(frame, f"{label_str}:{'ACT' if arm.active else 'idle'}"
                       f"{'[FRZ]' if arm.frozen else ''}",
                cx, H-20, 0.40, col)

        cv2.imshow("Gesture Data Collector", frame)
        cv2.waitKey(1)

        # ── 全局按键检测（不需要窗口焦点）──────────────
        if keyboard.is_pressed('r') and not _key_held.get('r'):
            _key_held['r'] = True
            if not recording:
                recording   = True
                current_seg = []
                print(f"[REC] 开始录制段 #{segment_id}")
            else:
                print("[REC] 请按 1 或 0 标注，或 ESC 丢弃")
        elif not keyboard.is_pressed('r'):
            _key_held['r'] = False

        if keyboard.is_pressed('1') and not _key_held.get('1') and recording:
            _key_held['1'] = True
            for row in current_seg:
                row["label"] = 1
                writer.writerow(row)
            stats[1] += len(current_seg)
            print(f"[SAVE] 段 #{segment_id} → 有意调节，{len(current_seg)//2} 帧")
            segment_id += 1
            recording   = False
            current_seg = []
            csv_file.flush()
        elif not keyboard.is_pressed('1'):
            _key_held['1'] = False

        if keyboard.is_pressed('0') and not _key_held.get('0') and recording:
            _key_held['0'] = True
            for row in current_seg:
                row["label"] = 0
                writer.writerow(row)
            stats[0] += len(current_seg)
            print(f"[SAVE] 段 #{segment_id} → 无意动作，{len(current_seg)//2} 帧")
            segment_id += 1
            recording   = False
            current_seg = []
            csv_file.flush()
        elif not keyboard.is_pressed('0'):
            _key_held['0'] = False

        if keyboard.is_pressed('esc') and not _key_held.get('esc'):
            _key_held['esc'] = True
            if recording:
                print(f"[DISCARD] 段 #{segment_id} 已丢弃（{len(current_seg)//2} 帧）")
                recording   = False
                current_seg = []
        elif not keyboard.is_pressed('esc'):
            _key_held['esc'] = False

        if keyboard.is_pressed('q') and not _key_held.get('q'):
            _key_held['q'] = True
            # 如果还在录制，自动丢弃未标注的段
            if recording and len(current_seg) > 0:
                print(f"[DISCARD] 退出时丢弃未标注段 #{segment_id}（{len(current_seg)//2} 帧）")
            break
        elif not keyboard.is_pressed('q'):
            _key_held['q'] = False

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    csv_file.close()
    print(f"\n✅ 采集完成，数据保存到 {OUTPUT_CSV}")
    print(f"   有意调节: {stats[1]} 帧")
    print(f"   无意动作: {stats[0]} 帧")
    print(f"   总段数:   {segment_id}")


if __name__ == "__main__":
    main()
