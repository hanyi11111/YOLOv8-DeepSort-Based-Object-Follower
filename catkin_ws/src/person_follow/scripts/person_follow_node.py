#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# 使用你已经在Jetson上跑通的ultralytics
from ultralytics import YOLO


class PersonFollowNode:
    def __init__(self):
        self.model_path = rospy.get_param("~model_path", "yolov8n.pt")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        self.conf = float(rospy.get_param("~conf", 0.35))
        self.target_area_ratio = float(rospy.get_param("~target_area_ratio", 0.10))  # 目标人框面积占比
        self.k_ang = float(rospy.get_param("~k_ang", 0.8))
        self.k_lin = float(rospy.get_param("~k_lin", 0.7))
        self.max_lin = float(rospy.get_param("~max_lin", 0.25))
        self.max_ang = float(rospy.get_param("~max_ang", 1.0))
        self.min_area_ratio = float(rospy.get_param("~min_area_ratio", 0.005))       # 太小当成无效目标
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 0.6))

        rospy.loginfo("Loading YOLO model: %s", self.model_path)
        self.model = YOLO(self.model_path)

        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)

        self.last_seen = rospy.Time(0)

    def image_to_bgr(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        enc = msg.encoding.lower()
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc == "mono8":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # bgr8 直接用
        return img

    def publish_stop(self):
        t = Twist()
        self.pub_cmd.publish(t)

    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def image_cb(self, msg):
        try:
            frame = self.image_to_bgr(msg)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", str(e))
            return

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        # 只检测 person (COCO class 0)
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=[0],
            verbose=False,
            device=0
        )[0]

        best = None
        best_area = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            for b in xyxy:
                x1, y1, x2, y2 = b[:4]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                area = bw * bh
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)

        cmd = Twist()

        if best is not None:
            x1, y1, x2, y2 = best
            cx = 0.5 * (x1 + x2)
            area_ratio = best_area / frame_area

            if area_ratio >= self.min_area_ratio:
                # 水平偏差 -> 转向
                ex = (cx - (w * 0.5)) / (w * 0.5)  # [-1, 1]
                cmd.angular.z = self.clamp(-self.k_ang * ex, -self.max_ang, self.max_ang)

                # 框面积偏差 -> 前进后退（面积越大表示越近）
                ed = self.target_area_ratio - area_ratio
                cmd.linear.x = self.clamp(self.k_lin * ed, -self.max_lin, self.max_lin)

                self.last_seen = rospy.Time.now()
            else:
                # 检到但太小，先停
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        else:
            # 没检到，超时则停
            if (rospy.Time.now() - self.last_seen).to_sec() > self.timeout_sec:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        self.pub_cmd.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("person_follow_node")
    node = PersonFollowNode()
    rospy.loginfo("person_follow_node started.")
    rospy.spin()
