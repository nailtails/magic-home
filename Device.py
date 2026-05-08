import cv2
import json
import numpy as np
from ultralytics import YOLO
import pykinect_azure as pykinect

# ==========================================
# 1. 所有可能的 YOLO 类别 -> 设备名 映射
#    这里保留未来会用到的 Device1 / Device2
# ==========================================
YOLO_CLASS_MAP = {
    "lamp": "Device1",
    "speaker": "Device2"
}

# ==========================================
# 2. 当前实际启用哪些设备
#    现在你只有音箱，就先只开 Device2
#    以后加上台灯时，改成 ["Device1", "Device2"] 即可
# ==========================================
ENABLED_DEVICES = ["Device1", "Device2"]

# 稳定性阈值 (单位：米)
STABILITY_THRESHOLD = 0.02

# 至少需要多少比例的有效帧
MIN_VALID_RATIO = 0.5


def check_stability(history_list, device_name):
    """校验坐标稳定性"""
    if len(history_list) == 0:
        return False

    data = np.array(history_list)
    std_dev = np.std(data, axis=0)

    if np.any(std_dev > STABILITY_THRESHOLD):
        print(f"   警告：{device_name} 在标定过程中发生移动！")
        print(f"   波动情况: X={std_dev[0]:.3f}m, Y={std_dev[1]:.3f}m, Z={std_dev[2]:.3f}m")
        return False

    return True


def extract_3d_position(mask, depth_img, fx, fy, cx, cy, img_w, img_h):
    """根据 mask 和深度图提取目标 3D 坐标"""
    raw_mask = cv2.resize(mask, (img_w, img_h))
    v_local, u_local = np.where(raw_mask > 0.5)
    if len(v_local) < 10:
        return None

    z_raw = depth_img[v_local, u_local] * 0.001  # mm -> m
    valid_mask = (z_raw > 0.1) & (z_raw < 4.0)
    z_valid = z_raw[valid_mask]
    if len(z_valid) < 5:
        return None

    z_base = np.percentile(z_valid, 20)
    fg_mask = z_valid < (z_base + 0.15)
    z_fg = z_valid[fg_mask]
    if len(z_fg) == 0:
        return None

    u_fg = u_local[valid_mask][fg_mask]
    v_fg = v_local[valid_mask][fg_mask]

    x_3d = (u_fg - cx) * z_fg / fx
    y_3d = (v_fg - cy) * z_fg / fy

    return [np.median(x_3d), np.median(y_3d), np.median(z_fg)]


def main():
    print(">>> [可配置标定模式] 正在启动相机与 YOLO 模型...")
    print(f">>> 当前启用设备: {ENABLED_DEVICES}")

    yolo_model = YOLO("best.pt")
    print(">>> YOLO 类别:", yolo_model.names)

    pykinect.initialize_libraries(track_body=False)
    kinect_cfg = pykinect.default_configuration
    kinect_cfg.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    kinect_cfg.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    kinect = pykinect.start_device(config=kinect_cfg)

    history = {device_name: [] for device_name in ENABLED_DEVICES}

    print("\n" + "=" * 50)
    print(">>> 请确保启用的设备都在画面中，并保持静止...")
    print(">>> 正在采集稳定坐标，请稍候...")
    print("=" * 50 + "\n")

    valid_frames = 0
    target_frames = 20

    try:
        while valid_frames < target_frames:
            capture = None
            try:
                capture = kinect.update()
                ret_c, color_img = capture.get_color_image()
                ret_d, depth_img = capture.get_transformed_depth_image()

                if not ret_c or not ret_d:
                    continue

                fx = kinect.calibration.color_params.fx
                fy = kinect.calibration.color_params.fy
                cx = kinect.calibration.color_params.cx
                cy = kinect.calibration.color_params.cy

                results = yolo_model(color_img, conf=0.5, verbose=False, retina_masks=True)[0]
                if results.boxes is None or results.masks is None:
                    continue

                boxes = results.boxes.cpu().numpy()
                masks = results.masks.data.cpu().numpy()
                h, w = color_img.shape[:2]

                detected_in_this_frame = set()

                for i, box in enumerate(boxes):
                    cls_name = results.names[int(box.cls[0])]

                    if cls_name not in YOLO_CLASS_MAP:
                        continue

                    mapped_name = YOLO_CLASS_MAP[cls_name]

                    if mapped_name not in ENABLED_DEVICES:
                        continue

                    coord = extract_3d_position(
                        masks[i], depth_img, fx, fy, cx, cy, w, h
                    )
                    if coord is None:
                        continue

                    history[mapped_name].append(coord)
                    detected_in_this_frame.add(mapped_name)

                valid_frames += 1

                status_text = []
                for dev in ENABLED_DEVICES:
                    count = len(history[dev])
                    mark = "√" if dev in detected_in_this_frame else "×"
                    status_text.append(f"{dev}:{mark}({count})")

                print(f"[{valid_frames}/{target_frames}] {' | '.join(status_text)}")

            finally:
                if capture is not None:
                    try:
                        capture.reset()   # 如果这里不行，再换成 capture.release_handle()
                    except Exception:
                        pass
                    capture = None

    finally:
        try:
            kinect.close()
        except Exception:
            pass

    min_valid_frames = target_frames * MIN_VALID_RATIO

    for dev in ENABLED_DEVICES:
        if len(history[dev]) < min_valid_frames:
            raise RuntimeError(f"标定失败：未能稳定识别到 {dev}！有效帧数不足。")

    for dev in ENABLED_DEVICES:
        is_stable = check_stability(history[dev], dev)
        if not is_stable:
            raise RuntimeError(f"标定失败：检测到 {dev} 在标定期间移动。")

    final_coords = {
        dev: np.mean(history[dev], axis=0).tolist()
        for dev in ENABLED_DEVICES
    }

    with open("device_positions.json", "w", encoding="utf-8") as f:
        json.dump(final_coords, f, indent=4, ensure_ascii=False)

    print(f"\n>>> 坐标稳定，已保存至 device_coords.json")
    for dev in ENABLED_DEVICES:
        x, y, z = final_coords[dev]
        print(f"   {dev}: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
    print()

if __name__ == "__main__":
    main()