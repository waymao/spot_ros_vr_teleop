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
        print("initialized.")

    

    def get_relative_pose(self, robot1, robot2):
        tf_robots = self.tf_buffer.lookup_transform(robot2 + "/body", robot1 + "/body", time=rospy.Time(0))
        q = tf_robots.transform.rotation
        quat = [q.x, q.y, q.z, q.w]

        # Convert to rotation matrix
        R = tf_trans.quaternion_matrix(quat)

        t = tf_trans.translation_matrix([tf_robots.transform.translation.x,
                                          tf_robots.transform.translation.y,
                                          tf_robots.transform.translation.z])

        return (R, t)


    # FIXME: please clean up
    def cmd_vel(self, vel: Twist):
        # transform wrt to the r
        time = rospy.Time().now()
        global_linear_vel = np.array([vel.linear.x, vel.linear.y, vel.linear.z])
        angular_speed = np.array([vel.angular.x, vel.angular.y, vel.angular.z])
        print("Received cmd_vel: linear={}, angular={}".format(global_linear_vel, angular_speed))

        for spot_name, spot_publisher in self.spot_vel_topics.items():
            R, t = self.get_relative_pose(self.base_spot_name, spot_name)
            radius = np.linalg.norm(t[0:3, 3])
            new_linear_vel = R[:3, :3] @ global_linear_vel
            
            linear_rotation_vel_magnitude = radius * angular_speed[2]
            linear_rotation_vel_vec  = np.array([t[1, 3], -t[0, 3], 0.0])
            if np.linalg.norm(linear_rotation_vel_vec) > 1e-6:
                linear_rotation_vel_vec /= np.linalg.norm(linear_rotation_vel_vec)
            else:
                linear_rotation_vel_vec = np.array([0.0, 0.0, 0.0])
            linear_rotation_vel_vec *= linear_rotation_vel_magnitude

            new_twist_cmd = Twist()
            total_linear_velocity = new_linear_vel + linear_rotation_vel_vec
            new_twist_cmd.linear.x = total_linear_velocity[0]
            new_twist_cmd.linear.y = total_linear_velocity[1]
            new_twist_cmd.linear.z = total_linear_velocity[2]
            new_twist_cmd.angular.x = angular_speed[0]
            new_twist_cmd.angular.y = angular_speed[1]
            new_twist_cmd.angular.z = angular_speed[2]
            print("{} linear speed: {}, angular speed: {}".format(
                spot_name, new_twist_cmd.linear, new_twist_cmd.angular)
            )
            spot_publisher.publish(new_twist_cmd)
        print()

if __name__ == "__main__":
    print("initializing...")
    rospy.init_node("spot_sync_drive")
    sync_drive = SpotSyncDrive(["spot", "spot2"])
    rospy.spin()
