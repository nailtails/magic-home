"""
视频录制脚本 record_video.py
==============================
录制手势视频片段，供后续标注使用。

按键说明（全局，不需要窗口焦点）：
  R  — 开始 / 停止录制一段
  Q  — 退出

视频保存到 ./videos/ 目录，文件名自动编号：
  clip_000.avi, clip_001.avi, ...
"""

import cv2
import os
import time
import keyboard

SAVE_DIR = "videos"
FPS      = 30
WIDTH    = 1280
HEIGHT   = 720


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 找下一个可用编号
    existing = [f for f in os.listdir(SAVE_DIR) if f.startswith("clip_") and f.endswith(".avi")]
    clip_idx = len(existing)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Record Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Record Video", 960, 540)

    recording  = False
    writer     = None
    clips_done = 0
    _key_held  = {}

    print("="*45)
    print("  视频录制")
    print("  R — 开始 / 停止录制")
    print("  Q — 退出")
    print(f"  视频保存到 ./{SAVE_DIR}/")
    print("="*45)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]

        if recording and writer is not None:
            writer.write(frame)

        # ── UI ────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (W, 70), (15, 15, 15), -1)
        cv2.rectangle(frame, (0, 0), (W, 70), (55, 55, 55),  1)

        if recording:
            # 闪烁红点
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (28, 28), 10, (0, 0, 220), -1)
            cv2.putText(frame, "REC", 45, 35, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (80, 160, 255), 2, cv2.LINE_AA) if False else None
            cv2.putText(frame, f"REC  {clip_name}   press R to stop",
                        (45, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 160, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"IDLE   saved: {clips_done} clips   press R to record / Q to quit",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Record Video", frame)
        cv2.waitKey(1)

        # ── 按键 ──────────────────────────────────────
        if keyboard.is_pressed('r') and not _key_held.get('r'):
            _key_held['r'] = True
            if not recording:
                clip_name = f"clip_{clip_idx:03d}.avi"
                clip_path = os.path.join(SAVE_DIR, clip_name)
                fourcc    = cv2.VideoWriter_fourcc(*"XVID")
                writer    = cv2.VideoWriter(clip_path, fourcc, FPS, (W, H))
                recording = True
                print(f"[REC] 开始录制 → {clip_path}")
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                print(f"[STOP] 已保存 clip_{clip_idx:03d}.avi")
                clips_done += 1
                clip_idx   += 1
        elif not keyboard.is_pressed('r'):
            _key_held['r'] = False

        if keyboard.is_pressed('q') and not _key_held.get('q'):
            _key_held['q'] = True
            if recording and writer:
                writer.release()
                print(f"[STOP] 退出时保存 clip_{clip_idx:03d}.avi")
                clips_done += 1
            break
        elif not keyboard.is_pressed('q'):
            _key_held['q'] = False

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 录制完成，共 {clips_done} 段视频保存在 ./{SAVE_DIR}/")


if __name__ == "__main__":
    main()
