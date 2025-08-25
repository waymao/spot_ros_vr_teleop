#!/usr/bin/env python3

import rospy
import tf2_ros
import tf.transformations as tf_trans
import numpy as np

from geometry_msgs.msg import Twist

class SpotSyncDrive:
    def __init__(self, spot_names):
        self.spot_vel_topics = {
            spot_name: rospy.Publisher(f"/{spot_name}/cmd_vel", Twist, queue_size=1)
            for spot_name in spot_names
        }
        self.base_spot_name = spot_names[0]
        # Reduce cache time - you're only using "now" transforms
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(0.5))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.vel_subscriber = rospy.Subscriber("/multi_spot/cmd_vel", Twist, callback=self.cmd_vel, queue_size=1)

    

    def get_rot_matrix(self, robot1, robot2):
        tf_robots = self.tf_buffer.lookup_transform(robot2 + "/body", robot1 + "/body", time=rospy.Time(0))
        q = tf_robots.transform.rotation
        quat = [q.x, q.y, q.z, q.w]

        # Convert to rotation matrix
        R = tf_trans.quaternion_matrix(quat)

        # This returns a 4x4 matrix; you might only want the top-left 3x3
        return R[:2, :2]


    def cmd_vel(self, vel: Twist):
        # transform wrt to the r
        time = rospy.Time().now()
        global_speed = np.array([vel.linear.x, vel.linear.y])
        for spot_name, spot_publisher in self.spot_vel_topics.items():
            rot_matrix = self.get_rot_matrix(self.base_spot_name, spot_name)
            curr_spot_speed = rot_matrix @ global_speed
            new_twist_cmd = Twist()
            new_twist_cmd.linear.x, new_twist_cmd.linear.y = curr_spot_speed.tolist()
            print("{} speed: {}".format(spot_name, curr_spot_speed))
            spot_publisher.publish(new_twist_cmd)
        print()

if __name__ == "__main__":
    rospy.init_node("spot_sync_drive")
    sync_drive = SpotSyncDrive(["spot", "spot2"])
    rospy.spin()
