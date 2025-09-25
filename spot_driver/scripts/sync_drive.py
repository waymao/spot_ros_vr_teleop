#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import tf.transformations as tf_trans
import numpy as np

from geometry_msgs.msg import Twist, TransformStamped

def tf_to_matrix(tf: TransformStamped):
    t = tf.transform.translation
    q = tf.transform.rotation
    quat = [q.x, q.y, q.z, q.w]
    R = np.eye(4)
    R[:3, :3] = tf_trans.quaternion_matrix(quat)[:3, :3]
    R[:3, 3] = [t.x, t.y, t.z]
    return R

def matrix_to_tf(R: np.ndarray, parent_frame: str, child_frame: str) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    translation = R[:3, 3]
    rotation = tf_trans.quaternion_from_matrix(R)
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]
    return t


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
        self.tf_publisher = tf2_ros.TransformBroadcaster()

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

        R_inv = np.linalg.inv(R)
        t_inv = -t

        return (R, t, R_inv, t_inv)
    
    def cmd_vel3( self, vel: Twist):

        # logic 
        # get relative pose between the two spots, gives a single vector p from base spot to other spot
        # calculate rotational velocity (vector crsos product)
        # distribute velocities to each spot since oppsoite in directions 

        if len(self.spot_vel_topics) != 2:
            rospy.logerr("Only support 2 spots for now")
            return
        
        base_spot_name = self.base_spot_name
        other_spot_name = [name for name in self.spot_vel_topics.keys() if name != base_spot_name][0]

        # use the odom frame of the base spot as the fixed world frame for calculations
        odom_frame = base_spot_name + "/odom"
        base_frame = base_spot_name + "/body"
        other_frame = other_spot_name + "/body"

        try:
            tf_odom_base = self.tf_buffer.lookup_transform(odom_frame, base_frame, time=rospy.Time(0), timeout=rospy.Duration(0.5))
            tf_odom_other = self.tf_buffer.lookup_transform(odom_frame, other_frame, time=rospy.Time(0), timeout=rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"Transform lookup failed: {e}")
            return
        
        # position of each robot in the odom frame
        p_base_in_odom = np.array([tf_odom_base.transform.translation.x,
                                   tf_odom_base.transform.translation.y,
                                   tf_odom_base.transform.translation.z])
        p_other_in_odom = np.array([tf_odom_other.transform.translation.x,
                                    tf_odom_other.transform.translation.y,
                                    tf_odom_other.transform.translation.z])
        
        # midpoint between the two robots in the odom frame
        p_mid_in_odom = (p_base_in_odom + p_other_in_odom) / 2.0

        # input velocity assumed to be in the odom frame
        v_mid_in_odom = np.array([vel.linear.x, vel.linear.y, vel.linear.z])
        omega_mid_in_odom = np.array([vel.angular.x, vel.angular.y, vel.angular.z])

        # calculate desired world velocity for each robot
        # vector from midpoint to base robot
        r_base_in_odom = p_base_in_odom - p_mid_in_odom
        # vecolity of base robot in odom frame 
        v_base_in_odom = v_mid_in_odom + np.cross(omega_mid_in_odom, r_base_in_odom)

        # vel from midpoint to other robot
        r_mid_other_in_odom = p_other_in_odom - p_mid_in_odom
        # velocity of other robot in odom frame
        v_other_in_odom = v_mid_in_odom + np.cross(omega_mid_in_odom, r_mid_other_in_odom)

        # transfrom world vel to body frames 
        q_odom_base = tf_odom_base.transform.rotation
        R_odom_base = tf_trans.quaternion_matrix([q_odom_base.x, q_odom_base.y, q_odom_base.z, q_odom_base.w])[:3, :3]
        R_base_odom = R_odom_base.T  # inverse rotation, transpose is inverse of rot 
        v_base_in_base = R_base_odom @ v_base_in_odom
        # transform angular vel to body frame to make robot face direction of travel
        omega_base_in_base = R_base_odom @ omega_mid_in_odom

        base_twist = Twist()
        base_twist.linear.x, base_twist.linear.y, base_twist.linear.z = v_base_in_base
        base_twist.angular.x, base_twist.angular.y, base_twist.angular.z = omega_base_in_base
        self.spot_vel_topics[base_spot_name].publish(base_twist)

        # for other spot 
        q_odom_other = tf_odom_other.transform.rotation
        R_odom_other = tf_trans.quaternion_matrix([q_odom_other.x, q_odom_other.y, q_odom_other.z, q_odom_other.w])[:3, :3]
        R_other_odom = R_odom_other.T
        v_other_in_other = R_other_odom @ v_other_in_odom
        # transform angular vel to body frame to make robot face direction of travel
        omega_other_in_other = R_other_odom @ omega_mid_in_odom

        other_twist = Twist()
        other_twist.linear.x, other_twist.linear.y, other_twist.linear.z = v_other_in_other
        other_twist.angular.x, other_twist.angular.y, other_twist.angular.z = omega_other_in_other
        self.spot_vel_topics[other_spot_name].publish(other_twist)

        rospy.loginfo_throttle(1.0,
                                f"Base {base_spot_name} cmd_vel: linear={base_twist.linear}, angular={base_twist.angular} | "
                                f"Other {other_spot_name} cmd_vel: linear={other_twist.linear}, angular={other_twist.angular}")


    # FIXME: please clean up
    def cmd_vel(self, vel: Twist):
        # transform wrt to the r
        time = rospy.Time().now()
        global_linear_vel = np.array([vel.linear.x, vel.linear.y, vel.linear.z])
        angular_speed = np.array([vel.angular.x, vel.angular.y, vel.angular.z])
        print("Received cmd_vel: linear={}, angular={}".format(global_linear_vel, angular_speed))
        assert(len(self.spot_vel_topics) == 2) # only support 2 spots

        # gather all robot transforms
        robot_tfs = {}
        robot_locs = {}
        for spot_name in self.spot_vel_topics.keys():
            spot_pose = self.tf_buffer.lookup_transform("gpe", f"{spot_name}/body", rospy.Time(0))
            robot_tfs[spot_name] = tf_to_matrix(spot_pose)
            robot_locs[spot_name] = robot_tfs[spot_name][:3, 3]

        # define the robot pivot as the midpoint between all robots
        # for now, the direction is relative to the base robot
        robot_pivot = np.array(list(robot_locs.values())).mean(axis=0)
        robot_pivot_R = np.eye(4)
        robot_pivot_R[:3, :3] = robot_tfs[self.base_spot_name][:3, :3]
        robot_pivot_R[:3, 3] = robot_pivot

        robot_pivot_tf = matrix_to_tf(robot_pivot_R, "gpe", "robot_pivot")
        self.tf_publisher.sendTransform(robot_pivot_tf)

        # Cap the angular speed to avoid instability
        max_angular_speed = 0.20
        if np.abs(angular_speed[2]) > max_angular_speed:
            angular_speed[2] = max_angular_speed * np.sign(angular_speed[2])
            
        # compute each spot v and send it out to each robot.
        for spot_name, spot_publisher in self.spot_vel_topics.items():
            is_base_spot = (spot_name == self.base_spot_name)

            relative_R = np.linalg.inv(robot_tfs[spot_name]) @ robot_pivot_R
            relative_xyz = relative_R[:3, 3]
            # print("relative_R for {}:\n{}".format(spot_name, relative_R))
            # print("relative_R extracted xyz for {}:\n{}".format(spot_name, relative_xyz))
            # print("diff in xyz for {}: {}".format(spot_name, robot_locs[spot_name] - robot_pivot))
            radius = np.linalg.norm(robot_locs[spot_name] - robot_pivot)  # Distance between robots and midpoint
            new_linear_vel = relative_R[:3, :3] @ global_linear_vel

            linear_rotation_vel_magnitude = radius * angular_speed[2]
            # the linear velocity vector that is perpendicular to the radius vector in the horizontal plane
            linear_rotation_vel_vec = np.array([relative_xyz[1], -relative_xyz[0], 0.0])
            if np.linalg.norm(linear_rotation_vel_vec) > 1e-6:
                linear_rotation_vel_vec /= np.linalg.norm(linear_rotation_vel_vec)
            else:
                linear_rotation_vel_vec = np.array([0.0, 0.0, 0.0])

            linear_rotation_vel_vec *= linear_rotation_vel_magnitude
            # Bump up the x component a bit, since the robots don't respond equally to x and y velocities
            linear_rotation_vel_vec[0] *= (1.0 + np.abs(linear_rotation_vel_vec[1] * 0.05))

            print("For {}, radius: {}, linear_rotation_vel_vec: {}".format(
                spot_name, radius, linear_rotation_vel_vec)
            )

            new_twist_cmd = Twist()
            new_linear_vel = new_linear_vel + linear_rotation_vel_vec
            new_twist_cmd.linear.x = new_linear_vel[0]
            new_twist_cmd.linear.y = new_linear_vel[1]
            new_twist_cmd.linear.z = new_linear_vel[2]
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
