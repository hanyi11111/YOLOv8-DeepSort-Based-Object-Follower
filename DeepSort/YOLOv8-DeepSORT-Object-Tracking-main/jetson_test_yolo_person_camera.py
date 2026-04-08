#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson 本地自检：从 ROS 订阅 CompressedImage，YOLO 只检 person(COCO 0)，在终端打印框与置信度。
不经过 rosbridge，用于确认「相机流 + 模型」能否在人前检出目标。

依赖：与 detector_client 相同环境（venv + ultralytics/torch），且已安装 Python3 的 rospy。
Melodic 若缺 rospy：sudo apt-get install ros-melodic-rospy python3-rospy

用法示例（不要给本脚本加内嵌 detect 的 PYTHONPATH，否则会加载带 DeepSORT 的 predict.py，
  未初始化 deepsort 时会报: NoneType has no attribute update）：
  source /opt/ros/melodic/setup.bash
  source ~/venv/deepsort-gpu/bin/activate
  python3 ~/jetson_test_yolo_person_camera.py \\
    --image-topic /camera/color/image_raw/compressed \\
    --weights yolov8n.pt --conf 0.45 --device 0
"""

from __future__ import print_function

import argparse
import contextlib
import logging
import os
import sys
import tempfile
import time

import cv2
import numpy as np


def _scrub_embedded_ultralytics_detect_from_path():
    """去掉 .../ultralytics/yolo/v8/detect，避免加载 DeepSORT 改过的 predict（deepsort 未初始化会崩）。"""
    for p in list(sys.path):
        if not p:
            continue
        n = p.replace("\\", "/")
        if "ultralytics/yolo/v8/detect" in n:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Subscribe CompressedImage, YOLO person-only, print detections."
    )
    p.add_argument(
        "--image-topic",
        default="/camera/color/image_raw/compressed",
        help="sensor_msgs/CompressedImage topic",
    )
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="min seconds between status lines (default 0.5)",
    )
    p.add_argument(
        "--every-frame",
        action="store_true",
        help="print a line for every processed frame (can be noisy)",
    )
    p.add_argument(
        "--keep-embedded-ultralytics-path",
        action="store_true",
        help="不清理 PYTHONPATH 里的 .../yolo/v8/detect（仅当你已修好内嵌 predict.py 时用）",
    )
    return p.parse_args()


def main():
    try:
        import rospy
        from sensor_msgs.msg import CompressedImage
    except ImportError as e:
        print(
            "Import rospy failed: {}\n"
            "Melodic 示例: sudo apt-get install ros-melodic-rospy python3-rospy".format(e),
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()
    if not args.keep_embedded_ultralytics_path:
        _scrub_embedded_ultralytics_detect_from_path()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(
            "Import ultralytics failed: {}\n"
            "若在 venv 里未装官方包可执行: pip install ultralytics".format(e),
            file=sys.stderr,
        )
        sys.exit(1)
    rospy.init_node("jetson_test_yolo_person_camera", anonymous=True)

    print(
        "Loading YOLO: {} device={} conf>={}".format(args.weights, args.device, args.conf),
        flush=True,
    )
    model = YOLO(args.weights)

    last_status = [0.0]
    frame_id = [0]

    def on_image(msg):
        if rospy.is_shutdown():
            return
        frame_id[0] += 1
        fid = frame_id[0]

        data = msg.data
        if hasattr(data, "tobytes"):
            buf = data.tobytes()
        else:
            buf = bytes(bytearray(data))
        if not buf:
            return

        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn_throttle(2.0, "cv2.imdecode failed")
            return

        h, w = frame.shape[:2]

        # 与 detector_client 一致：避免部分 8.0.3 对 ndarray 的误判，走临时 jpg
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            cv2.imwrite(tmp_path, frame)
            try:
                with _silence_ultralytics_predict_spam():
                    results = model.predict(
                        source=tmp_path,
                        conf=args.conf,
                        classes=[0],
                        verbose=False,
                        device=args.device,
                    )
            except AttributeError as ex:
                if "deepsort" in str(ex).lower() or "update" in str(ex):
                    rospy.logerr_throttle(
                        5.0,
                        "predict 走了内嵌 DeepSORT 版 ultralytics 且 deepsort 未初始化。"
                        "请去掉 PYTHONPATH 里的 .../ultralytics/yolo/v8/detect，"
                        "或 pip install ultralytics 后重试（不要用 --keep-embedded-ultralytics-path）。",
                    )
                raise
            if isinstance(results, (list, tuple)):
                if len(results) == 0:
                    result = None
                else:
                    result = results[0]
            else:
                result = results
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        now = time.time()
        n_person = 0
        lines = []
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            for i, (box, cf, ci) in enumerate(zip(xyxy, confs, clss)):
                if int(ci) != 0:
                    continue
                x1, y1, x2, y2 = box.tolist()
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if bw < 2 or bh < 2:
                    continue
                n_person += 1
                cx = 0.5 * (x1 + x2)
                lines.append(
                    "  #{} conf={:.3f} box=({:.0f},{:.0f})-({:.0f},{:.0f}) cx={:.0f} h={:.0f}".format(
                        i, float(cf), x1, y1, x2, y2, cx, bh
                    )
                )

        should_print = args.every_frame or (now - last_status[0] >= args.print_interval)
        if not should_print:
            return
        last_status[0] = now

        if n_person == 0:
            print(
                "[{}] frame={} size={}x{} NO person (class0 conf>={})".format(
                    time.strftime("%H:%M:%S"), fid, w, h, args.conf
                ),
                flush=True,
            )
        else:
            print(
                "[{}] frame={} size={}x{} PERSON x{}".format(
                    time.strftime("%H:%M:%S"), fid, w, h, n_person
                ),
                flush=True,
            )
            for ln in lines:
                print(ln, flush=True)

    sub = rospy.Subscriber(
        args.image_topic, CompressedImage, on_image, queue_size=1, buff_size=2 ** 24
    )
    print("Subscribed: {}".format(args.image_topic), flush=True)
    print("Ctrl+C to stop.", flush=True)
    rospy.spin()


if __name__ == "__main__":
    main()
