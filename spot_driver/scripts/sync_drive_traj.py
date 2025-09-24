#!/usr/bin/env python3

import rospy
import tf2_ros
import tf.transformations as tf_trans
import numpy as np

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Duration
from spot_msgs.msg import TrajectoryActionGoal, TrajectoryGoal
from nav_msgs.msg import Path

class SpotSyncDrive:
    def __init__(self, spot_names):
        self.spot_traj_topics = {
            spot_name: rospy.Publisher(f"/{spot_name}/trajectory/goal", TrajectoryActionGoal, queue_size=1)
            for spot_name in spot_names
        }
        self.spot_path_topics = {
            spot_name: rospy.Publisher(f"/{spot_name}/trajectory/path", Path, queue_size=1)
            for spot_name in spot_names
        }
        self.base_spot_name = spot_names[0]
        # Reduce cache time - you're only using "now" transforms
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(secs=0, nsecs=500000000))
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
        # Global velocity command in gpe (world) frame
        global_linear_vel = np.array([vel.linear.x, vel.linear.y, vel.linear.z])
        angular_speed = np.array([vel.angular.x, vel.angular.y, vel.angular.z])
        print("Received cmd_vel: linear={}, angular={}".format(global_linear_vel, angular_speed))

        for spot_name in self.spot_traj_topics.keys():
            spot_publisher = self.spot_traj_topics[spot_name]
            path_publisher = self.spot_path_topics[spot_name]
            try:
                # Transform global velocity to this robot's coordinate frame
                # First get transform from gpe to this robot's body frame
                gpe_to_body = self.tf_buffer.lookup_transform(f"{spot_name}/body", "gpe", rospy.Time(0))

                # Extract rotation matrix to transform velocities
                q = gpe_to_body.transform.rotation
                quat = [q.x, q.y, q.z, q.w]
                R = tf_trans.quaternion_matrix(quat)

                # Transform linear velocity to robot's body frame
                robot_linear_vel = R[:3, :3] @ global_linear_vel

                # For synchronized motion, we also need to account for the robot's position relative to formation center
                # Get position of this robot relative to base robot in gpe frame
                if spot_name != self.base_spot_name:
                    base_to_gpe = self.tf_buffer.lookup_transform("gpe", f"{self.base_spot_name}/body", rospy.Time(0))
                    robot_to_gpe = self.tf_buffer.lookup_transform("gpe", f"{spot_name}/body", rospy.Time(0))

                    # Calculate relative position vector in gpe frame
                    rel_pos = np.array([
                        robot_to_gpe.transform.translation.x - base_to_gpe.transform.translation.x,
                        robot_to_gpe.transform.translation.y - base_to_gpe.transform.translation.y,
                        0.0
                    ])

                    # Add rotational component for formation keeping
                    radius = np.linalg.norm(rel_pos)
                    if radius > 1e-6:
                        # Perpendicular vector for rotation
                        perp_vec = np.array([-rel_pos[1], rel_pos[0], 0.0]) / radius
                        rotational_vel = perp_vec * radius * angular_speed[2]

                        # Transform rotational velocity to robot frame
                        rotational_vel_robot = R[:3, :3] @ rotational_vel
                        robot_linear_vel += rotational_vel_robot

                # Now calculate target pose in this robot's odom frame
                dt = 0.1  # 100ms trajectory duration

                # Get current pose of the robot in its own odom frame
                current_transform = self.tf_buffer.lookup_transform(f"{spot_name}/odom", f"{spot_name}/body", rospy.Time(0))
                current_pos = current_transform.transform.translation
                current_rot = current_transform.transform.rotation

                # Calculate target position in robot's body frame, then transform to odom
                target_pos_body = robot_linear_vel * dt
                target_yaw_body = angular_speed[2] * dt

                # Get current orientation in odom frame
                current_yaw = tf_trans.euler_from_quaternion([current_rot.x, current_rot.y, current_rot.z, current_rot.w])[2]

                # Transform target position from body frame to odom frame
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                target_x = current_pos.x + (target_pos_body[0] * cos_yaw - target_pos_body[1] * sin_yaw)
                target_y = current_pos.y + (target_pos_body[0] * sin_yaw + target_pos_body[1] * cos_yaw)
                target_yaw = current_yaw + target_yaw_body
                target_quat = tf_trans.quaternion_from_euler(0, 0, target_yaw)

                # Create trajectory goal message
                traj_goal = TrajectoryActionGoal()
                traj_goal.header.stamp = rospy.Time.now()
                traj_goal.header.frame_id = f"{spot_name}/odom"

                # Create target pose in robot's odom frame
                traj_goal.goal.target_pose.header.stamp = rospy.Time.now()
                traj_goal.goal.target_pose.header.frame_id = f"{spot_name}/odom"
                traj_goal.goal.target_pose.pose.position.x = target_x
                traj_goal.goal.target_pose.pose.position.y = target_y
                traj_goal.goal.target_pose.pose.position.z = current_pos.z
                traj_goal.goal.target_pose.pose.orientation.x = target_quat[0]
                traj_goal.goal.target_pose.pose.orientation.y = target_quat[1]
                traj_goal.goal.target_pose.pose.orientation.z = target_quat[2]
                traj_goal.goal.target_pose.pose.orientation.w = target_quat[3]

                # Set duration and precise positioning
                traj_goal.goal.duration.data = rospy.Duration(secs=0, nsecs=int(dt * 1e9))
                traj_goal.goal.precise_positioning = False

                print("{} target pose: x={:.3f}, y={:.3f}, yaw={:.3f}".format(
                    spot_name, target_x, target_y, target_yaw)
                )
                spot_publisher.publish(traj_goal)

                # Create and publish smooth interpolated visualization path
                path_msg = Path()
                path_msg.header.stamp = rospy.Time.now()
                path_msg.header.frame_id = f"{spot_name}/odom"

                # Interpolate between current and target poses with 10 waypoints
                num_waypoints = 10
                for i in range(num_waypoints + 1):  # Include start and end
                    t = float(i) / num_waypoints  # Parameter from 0 to 1

                    # Linear interpolation for position
                    interp_x = current_pos.x + t * (target_x - current_pos.x)
                    interp_y = current_pos.y + t * (target_y - current_pos.y)
                    interp_z = current_pos.z  # Keep Z constant

                    # SLERP (Spherical Linear Interpolation) for orientation
                    current_quat = [current_rot.x, current_rot.y, current_rot.z, current_rot.w]

                    # Ensure shortest path by checking dot product
                    dot_product = (current_quat[0] * target_quat[0] +
                                 current_quat[1] * target_quat[1] +
                                 current_quat[2] * target_quat[2] +
                                 current_quat[3] * target_quat[3])

                    # If dot product is negative, negate one quaternion for shortest path
                    if dot_product < 0.0:
                        target_quat_adj = [-q for q in target_quat]
                    else:
                        target_quat_adj = target_quat

                    # Perform SLERP
                    if abs(dot_product) > 0.9995:  # Nearly identical orientations
                        # Use linear interpolation to avoid division by zero
                        interp_quat = [
                            current_quat[j] + t * (target_quat_adj[j] - current_quat[j])
                            for j in range(4)
                        ]
                        # Normalize
                        norm = np.sqrt(sum(q*q for q in interp_quat))
                        interp_quat = [q/norm for q in interp_quat]
                    else:
                        # Proper SLERP
                        omega = np.arccos(abs(dot_product))
                        sin_omega = np.sin(omega)
                        interp_quat = [
                            (np.sin((1-t) * omega) * current_quat[j] + np.sin(t * omega) * target_quat_adj[j]) / sin_omega
                            for j in range(4)
                        ]

                    # Create waypoint
                    waypoint = PoseStamped()
                    waypoint.header.stamp = rospy.Time.now()
                    waypoint.header.frame_id = f"{spot_name}/odom"
                    waypoint.pose.position.x = interp_x
                    waypoint.pose.position.y = interp_y
                    waypoint.pose.position.z = interp_z
                    waypoint.pose.orientation.x = interp_quat[0]
                    waypoint.pose.orientation.y = interp_quat[1]
                    waypoint.pose.orientation.z = interp_quat[2]
                    waypoint.pose.orientation.w = interp_quat[3]

                    path_msg.poses.append(waypoint)

                path_publisher.publish(path_msg)

            except Exception as e:
                rospy.logwarn(f"Failed to get transform for {spot_name}: {e}")
                continue
        print()

if __name__ == "__main__":
    print("initializing...")
    rospy.init_node("spot_sync_drive")
    sync_drive = SpotSyncDrive(["spot", "spot2"])
    rospy.spin()
