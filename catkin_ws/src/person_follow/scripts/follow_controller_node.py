#!/usr/bin/python2
# -*- coding: utf-8 -*-

import json
import time

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class FollowController(object):
    def __init__(self):
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.target_sub = rospy.Subscriber('/person_follow/target', String, self.target_cb, queue_size=1)

        self.last_target_time = 0.0

        self.image_width = float(rospy.get_param('~image_width', 1280.0))
        self.target_timeout = float(rospy.get_param('~target_timeout', 0.5))

        self.kp_ang = float(rospy.get_param('~kp_ang', 0.8))
        self.kp_lin = float(rospy.get_param('~kp_lin', 0.6))
        self.desired_h = float(rospy.get_param('~desired_h', 300.0))
        self.max_lin = float(rospy.get_param('~max_lin', 0.3))
        self.max_ang = float(rospy.get_param('~max_ang', 1.0))

    def _clip(self, v, low, high):
        return max(low, min(high, v))

    def target_cb(self, msg):
        try:
            data = json.loads(msg.data)

            cx = float(data.get('cx'))
            h = float(data.get('h'))
            _id = data.get('id', -1)

            if h <= 0.0:
                rospy.logwarn("invalid target h: %.3f", h)
                return

            self.last_target_time = time.time()

            err_x = (cx - self.image_width * 0.5) / (self.image_width * 0.5)
            err_d = (self.desired_h - h) / self.desired_h

            cmd = Twist()
            cmd.angular.z = self._clip(-self.kp_ang * err_x, -self.max_ang, self.max_ang)
            cmd.linear.x = self._clip(self.kp_lin * err_d, -self.max_lin, self.max_lin)

            self.cmd_pub.publish(cmd)
            rospy.loginfo_throttle(1.0, "track_id=%s cx=%.1f h=%.1f lin=%.2f ang=%.2f",
                                   str(_id), cx, h, cmd.linear.x, cmd.angular.z)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "target parse failed: %s", str(e))

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if time.time() - self.last_target_time > self.target_timeout:
                self.cmd_pub.publish(Twist())  # target timeout -> stop
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('follow_controller_node')
    node = FollowController()
    node.spin()
