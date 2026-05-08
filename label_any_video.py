
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import keyboard
from tkinter import Tk, filedialog
from collections import deque

# ── 参数（与主程序一致）──────────────────────────────
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

OUTPUT_CSV = "gesture_data.csv"
CSV_HEADER = [
    "frame", "amp_y", "vel_y", "delta_y", "delta_x",
    "weight", "active", "frozen", "label", "segment", "source"
]


# ════════════════════════════════════════════════════
#  特征提取（轻量版 ArmState）
# ════════════════════════════════════════════════════
class KF2D:
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
    def velocity(self): return float(self.kf.x[2]), float(self.kf.x[3])
    def reset(self): self._init = False; self.kf.P = np.eye(4) * 0.1


class ArmState:
    def __init__(self, side):
        self.side    = side
        self.active  = False
        self.frozen  = False
        self.kf      = KF2D()
        self.amp_y   = 0.0
        self.vel_y   = 0.0
        self._buf_y  = deque(maxlen=MOTION_WIN)
        self._buf_x  = deque(maxlen=MOTION_WIN)
        self.delta_y = 0.0
        self.delta_x = 0.0
        self.weight  = 0.0
        self._drop_streak     = 0
        self._deact_countdown = 0

    def update(self, shoulder, elbow, wrist, opp, hip, palm_pt=None):
        sw    = max(abs(shoulder[0] - opp[0]), 0.05)
        chest = 0.65 * shoulder + 0.35 * hip
        cx    = palm_pt[0] if palm_pt is not None else (wrist[0]+elbow[0])/2
        cy    = palm_pt[1] if palm_pt is not None else (wrist[1]+elbow[1])/2
        raw_y = (chest[1] - cy) / sw
        raw_x = ((cx - shoulder[0]) / sw) * (1 if self.side=="right" else -1)

        just_act = False
        if not self.active:
            self.amp_y = raw_y; self.vel_y = 0.0
            if raw_y > ACTIVATE_THRESH:
                self.active = True; just_act = True
                self.kf.reset(); self.kf.update(cx, cy)
                self.frozen = False; self._drop_streak = 0
        else:
            if raw_y < DEACTIVATE_THRESH:
                self._deact_countdown += 1
                self.frozen = True
                if self._deact_countdown >= 5:
                    self.active = False; self.frozen = False
                    self._drop_streak = 0; self._deact_countdown = 0
                    self._buf_y.clear(); self._buf_x.clear()
                self.amp_y = raw_y; return
            else:
                self._deact_countdown = 0
            fx, fy = self.kf.update(cx, cy)
            _, kvy = self.kf.velocity
            self.amp_y = (chest[1] - fy) / sw
            self.vel_y = -kvy / sw

        if self.active:
            self._buf_y.append(self.amp_y); self._buf_x.append(raw_x)
            if len(self._buf_y) >= MOTION_WIN:
                self.delta_y = self._buf_y[-1] - self._buf_y[0]
                self.delta_x = self._buf_x[-1] - self._buf_x[0]
            raw_w = 1.0 - MOTION_K * ((self.amp_y - MOTION_PEAK) ** 2)
            self.weight = max(MOTION_MIN_RATE, min(1.0, raw_w))
            if not just_act:
                if self.vel_y < -DROP_VEL_THRESH: self.frozen = True
                self._drop_streak = self._drop_streak+1 if self.vel_y < -MOTION_DEAD_ZONE else 0
                if self._drop_streak >= DROP_STREAK: self.frozen = True
                if self.vel_y > MOTION_DEAD_ZONE * 1.5:
                    self.frozen = False; self._drop_streak = 0

    def features(self):
        return {
            "amp_y":   round(self.amp_y,   4),
            "vel_y":   round(self.vel_y,   4),
            "delta_y": round(self.delta_y, 4),
            "delta_x": round(self.delta_x, 4),
            "weight":  round(self.weight,  4),
            "active":  int(self.active),
            "frozen":  int(self.frozen),
        }


def extract_features(frames, pose, hands, IDX):
    """对帧列表提取特征，返回 list[dict]"""
    mp_pose = mp.solutions.pose
    larm = ArmState("left")
    rarm = ArmState("right")
    out  = []
    for idx, frame in enumerate(frames):
        H, W  = frame.shape[:2]
        small = cv2.resize(frame, (W//2, H//2))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pr = pose.process(rgb)
        hr = hands.process(rgb)
        rgb.flags.writeable = True

        palm = {"Left": None, "Right": None}
        if hr.multi_hand_landmarks:
            for hlm, hi in zip(hr.multi_hand_landmarks, hr.multi_handedness):
                lbl = hi.classification[0].label
                palm[lbl] = np.array([
                    float(np.mean([hlm.landmark[i].x for i in PALM_BASE])),
                    float(np.mean([hlm.landmark[i].y for i in PALM_BASE]))
                ])

        if pr.pose_landmarks:
            lm = pr.pose_landmarks.landmark
            def gn(k): return np.array([lm[IDX[k]].x, lm[IDX[k]].y])
            ls,le,lw = gn("LS"),gn("LE"),gn("LW")
            rs,re,rw = gn("RS"),gn("RE"),gn("RW")
            lh,rh    = gn("LH"),gn("RH")
            larm.update(ls,le,lw,rs,lh, palm_pt=palm["Left"])
            rarm.update(rs,re,rw,ls,rh, palm_pt=palm["Right"])

        for arm in [larm, rarm]:
            f = arm.features(); f["frame"] = idx
            out.append(f)
    return out


# ════════════════════════════════════════════════════
#  UI 工具
# ════════════════════════════════════════════════════
def put(img, text, x, y, scale=0.45, color=(210,210,210), bold=False):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


def draw_timeline(frame, total, s, e, cur, W, H):
    if total < 2: return
    bx0, bx1, by = 20, W-20, H-22
    bw = bx1 - bx0
    cv2.rectangle(frame, (bx0, by-5), (bx1, by+5), (45,45,45), -1)
    if s is not None and e is not None:
        sx = bx0 + int(s/total*bw)
        ex = bx0 + int(e/total*bw)
        cv2.rectangle(frame, (sx, by-5), (ex, by+5), (0,150,255), -1)
    elif s is not None:
        sx = bx0 + int(s/total*bw)
        cv2.line(frame, (sx, by-10), (sx, by+10), (0,200,255), 2)
    cx = bx0 + int(cur/total*bw)
    cv2.line(frame, (cx, by-12), (cx, by+12), (255,255,255), 2)
    put(frame, f"{cur}/{total}", bx0, by-14, 0.32, (130,130,130))


def pick_files():
    """弹出文件选择框，返回选中的文件路径列表"""
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    paths = filedialog.askopenfilenames(
        title="选择视频文件（可多选）",
        filetypes=[("视频文件", "*.avi *.mp4 *.mov *.mkv"), ("所有文件", "*.*")]
    )
    root.destroy()
    return list(paths)


# ════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════
def main():
    # ── 选择视频 ────────────────────────────────────
    print("请在弹出的文件选择框中选择视频文件（可多选）...")
    clip_queue = pick_files()
    if not clip_queue:
        print("未选择任何文件，退出")
        return
    print(f"已选择 {len(clip_queue)} 个文件")

    # ── MediaPipe ───────────────────────────────────
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose  = mp_pose.Pose(model_complexity=0,
                         min_detection_confidence=0.6,
                         min_tracking_confidence=0.6)
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)
    PL  = mp_pose.PoseLandmark
    IDX = {
        "LS": PL.LEFT_SHOULDER,  "LE": PL.LEFT_ELBOW,  "LW": PL.LEFT_WRIST,
        "RS": PL.RIGHT_SHOULDER, "RE": PL.RIGHT_ELBOW, "RW": PL.RIGHT_WRIST,
        "LH": PL.LEFT_HIP,       "RH": PL.RIGHT_HIP,
    }

    # ── CSV ─────────────────────────────────────────
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file    = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    csv_writer  = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if not file_exists:
        csv_writer.writeheader()

    cv2.namedWindow("Label Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Video", 960, 540)

    segment_id = 0
    stats      = {1: 0, 0: 0}
    _kh        = {}   # key_held
    quit_all   = False

    def pressed(k):
        if keyboard.is_pressed(k) and not _kh.get(k):
            _kh[k] = True; return True
        if not keyboard.is_pressed(k): _kh[k] = False
        return False

    print("\n" + "="*45)
    print("  视频标注工具（自选文件版）")
    print("  空格  — 播放/暂停    ←/→ — 跳帧(5帧)")
    print("  S     — 设起点       E   — 设终点")
    print("  1     — 有意调节     0   — 无意动作")
    print("  C     — 清除选择     A   — 添加视频")
    print("  N     — 下一个       Q   — 退出保存")
    print("="*45)

    clip_idx = 0
    while clip_idx < len(clip_queue) and not quit_all:
        clip_path = clip_queue[clip_idx]

        # ── 加载视频帧 ──────────────────────────────
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"⚠️  无法打开 {clip_path}，跳过")
            clip_idx += 1; continue

        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"\n[LOAD] 加载 {os.path.basename(clip_path)}...")
        all_frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            all_frames.append(cv2.flip(f, 1))
        cap.release()
        total = len(all_frames)
        print(f"[LOAD] {total} 帧 @ {fps_vid:.0f}fps")

        if total == 0:
            print("⚠️  视频为空，跳过")
            clip_idx += 1; continue

        cv2.createTrackbar("Frame", "Label Video", 0, total-1, lambda x: None)

        cur      = 0
        playing  = False
        seg_s    = None
        seg_e    = None
        skip     = False

        while not quit_all:
            # 播放推进
            if playing:
                cur = min(cur + 1, total - 1)
                cv2.setTrackbarPos("Frame", "Label Video", cur)
                if cur >= total - 1: playing = False
            else:
                tb = cv2.getTrackbarPos("Frame", "Label Video")
                if tb != cur: cur = tb

            frame = all_frames[cur].copy()
            H, W  = frame.shape[:2]

            # ── 绘制 UI ─────────────────────────────
            cv2.rectangle(frame, (0,0), (W,85), (12,12,12), -1)
            cv2.rectangle(frame, (0,0), (W,85), (50,50,50),  1)

            fname = os.path.basename(clip_path)
            put(frame, f"{fname}  [{clip_idx+1}/{len(clip_queue)}]  frame {cur}/{total-1}",
                12, 22, 0.42, (170,170,170))

            st_col = (80,220,80) if playing else (130,130,130)
            put(frame, "▶ PLAY" if playing else "⏸ PAUSE", 12, 48, 0.40, st_col)

            if seg_s is not None and seg_e is not None:
                put(frame, f"SEL {seg_s}→{seg_e} ({seg_e-seg_s}f)  "
                           f"press 1=intentional  0=unintentional  C=clear",
                    90, 48, 0.36, (0,200,255))
            elif seg_s is not None:
                put(frame, f"start={seg_s}   press E to set end point",
                    90, 48, 0.36, (255,180,0))
            else:
                put(frame, "S=start  E=end  A=add video  N=next  Q=quit",
                    90, 48, 0.36, (110,110,110))

            put(frame,
                f"saved: intent={stats[1]}  unintent={stats[0]}  segs={segment_id}  "
                f"out={OUTPUT_CSV}",
                12, 70, 0.32, (70,170,70))

            draw_timeline(frame, total-1, seg_s, seg_e, cur, W, H)
            cv2.imshow("Label Video", frame)
            cv2.waitKey(max(1, int(1000/fps_vid)) if playing else 25)

            # ── 按键处理 ────────────────────────────
            if pressed('space'): playing = not playing

            if pressed('right'):
                cur = min(cur+5, total-1)
                cv2.setTrackbarPos("Frame", "Label Video", cur)

            if pressed('left'):
                cur = max(cur-5, 0)
                cv2.setTrackbarPos("Frame", "Label Video", cur)

            if pressed('s'):
                seg_s = cur; seg_e = None
                print(f"[S] 起点 = 帧 {seg_s}")

            if pressed('e'):
                if seg_s is None:
                    print("[E] 请先按 S 设置起点")
                else:
                    seg_e = cur
                    if seg_e < seg_s: seg_s, seg_e = seg_e, seg_s
                    print(f"[E] 终点 = 帧 {seg_e}，共 {seg_e-seg_s} 帧")

            if pressed('c'):
                seg_s = None; seg_e = None
                print("[C] 已清除选择")

            # 标注：1 或 0
            for key_ch, lval in [('1',1), ('0',0)]:
                if pressed(key_ch):
                    if seg_s is None or seg_e is None:
                        print(f"[{key_ch}] 请先用 S / E 选定片段范围")
                    else:
                        lname = "有意调节" if lval==1 else "无意动作"
                        seg_frames = all_frames[seg_s:seg_e+1]
                        print(f"[{key_ch}] 提取 {len(seg_frames)} 帧特征中...")
                        feats = extract_features(seg_frames, pose, hands, IDX)
                        for feat in feats:
                            feat.update({
                                "label":   lval,
                                "segment": segment_id,
                                "source":  os.path.basename(clip_path),
                            })
                            csv_writer.writerow(feat)
                        csv_file.flush()
                        stats[lval] += len(feats)
                        print(f"[SAVE] 段#{segment_id} → {lname}，{len(feats)} 帧")
                        segment_id += 1
                        seg_s = None; seg_e = None

            if pressed('a'):
                print("请在弹出的文件选择框中选择更多视频...")
                new_files = pick_files()
                if new_files:
                    clip_queue.extend(new_files)
                    print(f"[A] 新增 {len(new_files)} 个文件，队列共 {len(clip_queue)} 个")

            if pressed('n'):
                print(f"[N] 跳到下一个视频")
                skip = True; break

            if pressed('q'):
                quit_all = True; break

        clip_idx += 1

    cv2.destroyAllWindows()
    pose.close(); hands.close()
    csv_file.close()

    print(f"\n✅ 标注完成")
    print(f"   有意调节 : {stats[1]} 帧")
    print(f"   无意动作 : {stats[0]} 帧")
    print(f"   总段数   : {segment_id}")
    print(f"   数据文件 : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
