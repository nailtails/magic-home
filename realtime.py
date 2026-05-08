import os
import time

import cv2

from config import Config
from predictor_core import RealtimeAttentionPredictor
from runtime_utils import KinectRuntime, draw_attention_overlay, draw_prob_overlay


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

    # 两个都不存在时，默认返回新名字，方便报错时更直观
    return "device_positions.json"


def main():
    cfg = Config()

    model_path = "lstm_best_model.pth"
    device_json_path = resolve_device_json_path()

    predictor = RealtimeAttentionPredictor(
        model_path=model_path,
        device_json_path=device_json_path,
        ema_alpha=0.5,
        use_ema=True,
        prob_window=5,
        conf_thresh=0.60,
        switch_patience=3,
        display_hold_count=3,
    )

    kinect_runtime = KinectRuntime()

    print("=" * 80)
    print("Azure Kinect Body Tracking + Attention Monitor")
    print(f"model_path        = {model_path}")
    print(f"device_json_path  = {device_json_path}")
    print(f"window_size       = {predictor.window_size}")
    print(f"stride            = {predictor.stride}")
    print(f"use_ema           = {predictor.use_ema}")
    print(f"runtime_device    = {predictor.device}")
    print("loaded model kwargs:")
    for k, v in predictor.model_kwargs.items():
        print(f"  {k:<16} = {v}")
    print("Press q to quit")
    print("=" * 80)

    try:
        while True:
            frame_data = kinect_runtime.get_live_frame_data()

            if frame_data is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.01)
                continue

            vis = frame_data["color_image"]
            joints_xyz = frame_data["joints_xyz"]
            has_body = frame_data["has_body"]

            result = None
            if joints_xyz is not None:
                try:
                    result = predictor.update(joints_xyz)
                except Exception as e:
                    print(f"[ERROR] predictor.update failed: {e}")

            vis = draw_attention_overlay(vis, result, has_body)
            vis = draw_prob_overlay(vis, result)

            cv2.imshow("Azure Kinect Body Tracking + Attention", vis)

            if result is not None:
                raw_prob = result["raw_prob"]
                avg_prob = result["avg_prob"]
                print(
                    f"frame={result['frame_count']:04d} | "
                    f"raw={result['raw_name']} "
                    f"[{raw_prob[0]:.3f}, {raw_prob[1]:.3f}, {raw_prob[2]:.3f}] | "
                    f"stable={result['stable_name']} "
                    f"[{avg_prob[0]:.3f}, {avg_prob[1]:.3f}, {avg_prob[2]:.3f}] | "
                    f"display={result['display_name']}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        kinect_runtime.close()


if __name__ == "__main__":
    main()