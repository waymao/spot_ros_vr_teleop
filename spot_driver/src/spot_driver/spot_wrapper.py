import time
import math
import numpy as np
import cv2
from std_msgs.msg import Float32MultiArray

from bosdyn.client import create_standard_sdk, ResponseError, RpcError
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.geometry import EulerZXY

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.frame_helpers import *
from bosdyn.client.power import safe_power_off, PowerClient, power_on
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.api.graph_nav import nav_pb2
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client import power
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
from bosdyn.client.exceptions import InternalServerError
from geometry_msgs.msg import Pose, Point, Quaternion

from . import graph_nav_util

import bosdyn.api.robot_state_pb2 as robot_state_proto
from bosdyn.api import basic_command_pb2, arm_command_pb2, manipulation_api_pb2, robot_command_pb2, synchronized_command_pb2, mobility_command_pb2
from google.protobuf.timestamp_pb2 import Timestamp

########
# Janeth edits for IK api call to bosdyn
import bosdyn.client.util
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.spot.inverse_kinematics_pb2 import (InverseKinematicsRequest,
                                                    InverseKinematicsResponse)
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME,
                                         GROUND_PLANE_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b)
from bosdyn.client.inverse_kinematics import InverseKinematicsClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.util import seconds_to_duration
########

def make_robot_command(arm_joint_traj): #UNDERGRADS (6-20-23 11am)
    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

front_image_sources = ['frontleft_fisheye_image', 'frontright_fisheye_image', 'frontleft_depth_in_visual_frame', 'frontright_depth_in_visual_frame']
#front_image_sources = ['frontleft_visual_in_depth_frame', 'frontright_visual_in_depth_frame', 'frontleft_depth', 'frontright_depth']
"""List of image sources for front image periodic query"""
side_image_sources = ['left_fisheye_image', 'right_fisheye_image', 'left_depth_in_visual_frame', 'right_depth_in_visual_frame']
#side_image_sources = ['left_visual_in_depth_frame', 'right_visual_in_depth_frame', 'left_depth', 'right_depth']
"""List of image sources for side image periodic query"""
rear_image_sources = ['back_fisheye_image', 'back_depth_in_visual_frame']
#rear_image_sources = ['back_visual_in_depth_frame', 'back_depth']
"""List of image sources for rear image periodic query"""

hand_image_sources = ['hand_color_image', 'hand_depth_in_hand_color_frame']

class AsyncRobotState(AsyncPeriodicQuery):
    """Class to get robot state at regular intervals.  get_robot_state_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncRobotState, self).__init__("robot-state", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_robot_state_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncMetrics(AsyncPeriodicQuery):
    """Class to get robot metrics at regular intervals.  get_robot_metrics_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncMetrics, self).__init__("robot-metrics", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_robot_metrics_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncLease(AsyncPeriodicQuery):
    """Class to get lease state at regular intervals.  list_leases_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncLease, self).__init__("lease", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.list_leases_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncImageService(AsyncPeriodicQuery):
    """Class to get images at regular intervals.  get_image_from_sources_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback, image_requests):
        super(AsyncImageService, self).__init__("robot_image_service", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback
        self._image_requests = image_requests

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_image_async(self._image_requests)
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncIdle(AsyncPeriodicQuery):
    """Class to check if the robot is moving, and if not, command a stand with the set mobility parameters

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            spot_wrapper: A handle to the wrapper library
    """
    def __init__(self, client, logger, rate, spot_wrapper):
        super(AsyncIdle, self).__init__("idle", client, logger,
                                           period_sec=1.0/rate)

        self._spot_wrapper = spot_wrapper

    def _start_query(self):
        if self._spot_wrapper._last_stand_command != None:
            try:
                response = self._client.robot_command_feedback(self._spot_wrapper._last_stand_command)
                self._spot_wrapper._is_sitting = False
                if (response.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status ==
                        basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING):
                    self._spot_wrapper._is_standing = True
                    self._spot_wrapper._last_stand_command = None
                else:
                    self._spot_wrapper._is_standing = False
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_stand_command = None

        if self._spot_wrapper._last_sit_command != None:
            try:
                self._spot_wrapper._is_standing = False
                response = self._client.robot_command_feedback(self._spot_wrapper._last_sit_command)
                if (response.feedback.synchronized_feedback.mobility_command_feedback.sit_feedback.status ==
                        basic_command_pb2.SitCommand.Feedback.STATUS_IS_SITTING):
                    self._spot_wrapper._is_sitting = True
                    self._spot_wrapper._last_sit_command = None
                else:
                    self._spot_wrapper._is_sitting = False
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_sit_command = None

        is_moving = False

        if self._spot_wrapper._last_velocity_command_time != None:
            if time.time() < self._spot_wrapper._last_velocity_command_time:
                is_moving = True
            else:
                self._spot_wrapper._last_velocity_command_time = None

        if self._spot_wrapper._last_arm_command_time != None:
            if time.time() < self._spot_wrapper._last_arm_command_time:
                is_moving = True
            else:
                self._spot_wrapper._last_arm_command_time = None

        if self._spot_wrapper._last_trajectory_command != None:
            try:
                response = self._client.robot_command_feedback(self._spot_wrapper._last_trajectory_command)
                status = response.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
                # STATUS_AT_GOAL always means that the robot reached the goal. If the trajectory command did not
                # request precise positioning, then STATUS_NEAR_GOAL also counts as reaching the goal
                if status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL or \
                    (status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL and
                     not self._spot_wrapper._last_trajectory_command_precise):
                    self._spot_wrapper._at_goal = True
                    # Clear the command once at the goal
                    self._spot_wrapper._last_trajectory_command = None
                elif status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_GOING_TO_GOAL:
                    is_moving = True
                elif status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL:
                    is_moving = True
                    self._spot_wrapper._near_goal = True
                else:
                    self._spot_wrapper._last_trajectory_command = None
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_trajectory_command = None

        self._spot_wrapper._is_moving = is_moving

        if self._spot_wrapper.is_standing and not self._spot_wrapper.is_moving:
            self._spot_wrapper.stand(False)

class SpotWrapper():
    """Generic wrapper class to encompass release 1.1.4 API features as well as maintaining leases automatically"""
    def __init__(self, username, password, hostname, logger, estop_timeout=9.0, rates = {}, callbacks = {}, cameras_used=[]):
        """
        Custom paramters (zkytony):
           camera_used (list): a list of strings among "front", "side", "rear" for which we will call image services.
               If empty, then no image service will be called here.
        """
        self._username = username
        self._password = password
        self._hostname = hostname
        self._logger = logger
        self._rates = rates
        self._callbacks = callbacks
        self._estop_timeout = estop_timeout
        self._keep_alive = True
        self._valid = True

        self._mobility_params = RobotCommandBuilder.mobility_params()
        self._is_standing = False
        self._is_sitting = True
        self._is_moving = False
        self._at_goal = False
        self._near_goal = False
        self._last_stand_command = None
        self._last_sit_command = None
        self._last_trajectory_command = None
        self._last_trajectory_command_precise = None
        self._last_velocity_command_time = None
        self._last_arm_command_time = None

        self._front_image_requests = []
        self._side_image_requests = []
        self._rear_image_requests = []
        self._hand_image_requests = []

        self._cameras_used = cameras_used

        if "front" in cameras_used:
            for source in front_image_sources:
                self._front_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        '''
        if "side" in cameras_used:
            for source in side_image_sources:
                self._side_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        if "rear" in cameras_used:
            for source in rear_image_sources:
                self._rear_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        if "hand" in cameras_used:
            for source in hand_image_sources:
                self._hand_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))
        '''

        try:
            self._sdk = create_standard_sdk('ros_spot')
        except Exception as e:
            self._logger.error("Error creating SDK object: %s", e)
            self._valid = False
            return

        self._robot = self._sdk.create_robot(self._hostname)

        try:
            self._robot.authenticate(self._username, self._password)
            self._robot.start_time_sync()
        except RpcError as err:
            self._logger.error("Failed to communicate with robot: %s", err)
            self._valid = False
            return

        if self._robot:
            # Clients
            try:
                self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
                self._robot_command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
                self._manipulation_client = self._robot.ensure_client(ManipulationApiClient.default_service_name)
                self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)
                self._power_client = self._robot.ensure_client(PowerClient.default_service_name)
                self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
                self._lease_wallet = self._lease_client.lease_wallet
                self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
                self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            except Exception as e:
                self._logger.error("Unable to create client service: %s", e)
                self._valid = False
                return

            # Store the most recent knowledge of the state of the robot based on rpc calls.
            self._current_graph = None
            self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
            self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
            self._current_edge_snapshots = dict()  # maps id to edge snapshot
            self._current_annotation_name_to_wp_id = dict()

            # Async Tasks
            self._async_task_list = []
            self._robot_state_task = AsyncRobotState(self._robot_state_client, self._logger, max(0.0, self._rates.get("robot_state", 0.0)), self._callbacks.get("robot_state", lambda:None))
            self._robot_metrics_task = AsyncMetrics(self._robot_state_client, self._logger, max(0.0, self._rates.get("metrics", 0.0)), self._callbacks.get("metrics", lambda:None))
            self._lease_task = AsyncLease(self._lease_client, self._logger, max(0.0, self._rates.get("lease", 0.0)), self._callbacks.get("lease", lambda:None))

            '''
            if "front" in self._cameras_used:
                self._front_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("front_image", 0.0)), self._callbacks.get("front_image", lambda:None), self._front_image_requests)
            if "side" in self._cameras_used:
                self._side_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("side_image", 0.0)), self._callbacks.get("side_image", lambda:None), self._side_image_requests)
            if "rear" in self._cameras_used:
                self._rear_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("rear_image", 0.0)), self._callbacks.get("rear_image", lambda:None), self._rear_image_requests)
            if "hand" in self._cameras_used:
                self._hand_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("hand_image", 0.0)), self._callbacks.get("rear_image", lambda:None), self._hand_image_requests)
            self._idle_task = AsyncIdle(self._robot_command_client, self._logger, 10.0, self)
            '''

            self._estop_endpoint = None

            tasks = [self._robot_state_task, self._robot_metrics_task, self._lease_task] #, self._idle_task]
            '''
            if "front" in self._cameras_used:
                tasks.append(self._front_image_task)
            if "side" in self._cameras_used:
                tasks.append(self._side_image_task)
            if "rear" in self._cameras_used:
                tasks.append(self._rear_image_task)
            if "hand" in self._cameras_used:
                tasks.append(self._hand_image_task)
            '''
            self._async_tasks = AsyncTasks(tasks)

            self._robot_id = None
            self._lease = None

    @property
    def logger(self):
        """Return logger instance of the SpotWrapper"""
        return self._logger

    @property
    def is_valid(self):
        """Return boolean indicating if the wrapper initialized successfully"""
        return self._valid

    @property
    def id(self):
        """Return robot's ID"""
        return self._robot_id

    @property
    def robot_state(self):
        """Return latest proto from the _robot_state_task"""
        return self._robot_state_task.proto

    @property
    def metrics(self):
        """Return latest proto from the _robot_metrics_task"""
        return self._robot_metrics_task.proto

    @property
    def lease(self):
        """Return latest proto from the _lease_task"""
        return self._lease_task.proto

    @property
    def front_images(self):
        """Return latest proto from the _front_image_task"""
        return self._front_image_task.proto

    @property
    def side_images(self):
        """Return latest proto from the _side_image_task"""
        return self._side_image_task.proto

    @property
    def rear_images(self):
        """Return latest proto from the _rear_image_task"""
        return self._rear_image_task.proto

    @property
    def is_standing(self):
        """Return boolean of standing state"""
        return self._is_standing

    @property
    def is_sitting(self):
        """Return boolean of standing state"""
        return self._is_sitting

    @property
    def is_moving(self):
        """Return boolean of walking state"""
        return self._is_moving

    @property
    def near_goal(self):
        return self._near_goal

    @property
    def at_goal(self):
        return self._at_goal

    @property
    def time_skew(self):
        """Return the time skew between local and spot time"""
        return self._robot.time_sync.endpoint.clock_skew

    def resetMobilityParams(self):
        """
        Resets the mobility parameters used for motion commands to the default values provided by the bosdyn api.
        Returns:
        """
        self._mobility_params = RobotCommandBuilder.mobility_params()

    def robotToLocalTime(self, timestamp):
        """Takes a timestamp and an estimated skew and return seconds and nano seconds in local time

        Args:
            timestamp: google.protobuf.Timestamp
        Returns:
            google.protobuf.Timestamp
        """

        rtime = Timestamp()

        rtime.seconds = timestamp.seconds - self.time_skew.seconds
        rtime.nanos = timestamp.nanos - self.time_skew.nanos
        if rtime.nanos < 0:
            rtime.nanos = rtime.nanos + 1000000000
            rtime.seconds = rtime.seconds - 1

        # Workaround for timestamps being incomplete
        if rtime.seconds < 0:
            rtime.seconds = 0
            rtime.nanos = 0

        return rtime

    def claim(self, take_lease=False):
        """Get a lease for the robot, a handle on the estop endpoint, and the ID of the robot."""
        try:
            self._robot_id = self._robot.get_id()
            self.getLease(take_lease=take_lease)
            self.resetEStop()
            return True, "Success"
        except (ResponseError, RpcError) as err:
            self._logger.error("Failed to initialize robot communication: %s", err)
            return False, str(err)

    def updateTasks(self):
        """Loop through all periodic tasks and update their data if needed."""
        try:
            from time import time
            if 'last_time' not in globals(): global last_time 
            else: pass#(time()-last_time)
            last_time = time()
            self._async_tasks.update()
        except Exception as e:
            print(f"Update tasks failed with error: {str(e)}")

    def resetEStop(self):
        """Get keepalive for eStop"""
        self._estop_endpoint = EstopEndpoint(self._estop_client, 'ros', self._estop_timeout)
        self._estop_endpoint.force_simple_setup()  # Set this endpoint as the robot's sole estop.
        self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)

    def assertEStop(self, severe=True):
        """Forces the robot into eStop state.

        Args:
            severe: Default True - If true, will cut motor power immediately.  If false, will try to settle the robot on the ground first
        """
        try:
            if severe:
                self._estop_keepalive.stop()
            else:
                self._estop_keepalive.settle_then_cut()

            return True, "Success"
        except:
            return False, "Error"

    def disengageEStop(self):
        """Disengages the E-Stop"""
        try:
            self._estop_keepalive.allow()
            return True, "Success"
        except:
            return False, "Error"


    def releaseEStop(self):
        """Stop eStop keepalive"""
        if self._estop_keepalive:
            self._estop_keepalive.stop()
            self._estop_keepalive = None
            self._estop_endpoint = None

    def getLease(self, take_lease=False):
        """Get a lease for the robot and keep the lease alive automatically."""
        if take_lease:
            self._lease = self._lease_client.take()
        else:
            self._lease = self._lease_client.acquire()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

    def releaseLease(self):
        """Return the lease on the body."""
        if self._lease:
            self._lease_client.return_lease(self._lease)
            self._lease = None

    def release(self):
        """Return the lease on the body and the eStop handle."""
        try:
            self.releaseLease()
            self.releaseEStop()
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        """Release control of robot as gracefully as posssible."""
        if self._robot.time_sync:
            self._robot.time_sync.stop()
        self.releaseLease()
        self.releaseEStop()

    def _robot_command(self, command_proto, end_time_secs=None, timesync_endpoint=None):
        """Generic blocking function for sending commands to robots.

        Args:
            command_proto: robot_command_pb2 object to send to the robot.  Usually made with RobotCommandBuilder
            end_time_secs: (optional) Time-to-live for the command in seconds
            timesync_endpoint: (optional) Time sync endpoint
        """
        try:
            id = self._robot_command_client.robot_command(lease=None, command=command_proto, end_time_secs=end_time_secs, timesync_endpoint=timesync_endpoint)
            return True, "Success", id
        except Exception as e:
            return False, str(e), None

    def stop(self):
        """Stop the robot's motion."""
        response = self._robot_command(RobotCommandBuilder.stop_command())
        return response[0], response[1]

    def self_right(self):
        """Have the robot self-right itself."""
        response = self._robot_command(RobotCommandBuilder.selfright_command())
        return response[0], response[1]

    def sit(self):
        """Stop the robot's motion and sit down if able."""
        response = self._robot_command(RobotCommandBuilder.synchro_sit_command())
        self._last_sit_command = response[2]
        return response[0], response[1]

    def stand(self, monitor_command=True):
        """If the e-stop is enabled, and the motor power is enabled, stand the robot up."""
        response = self._robot_command(RobotCommandBuilder.synchro_stand_command(params=self._mobility_params))
        if monitor_command:
            self._last_stand_command = response[2]
        return response[0], response[1]

    def safe_power_off(self):
        """Stop the robot's motion and sit if possible.  Once sitting, disable motor power."""
        response = self._robot_command(RobotCommandBuilder.safe_power_off_command())
        return response[0], response[1]

    def clear_behavior_fault(self, id):
        """Clear the behavior fault defined by id."""
        try:
            rid = self._robot_command_client.clear_behavior_fault(behavior_fault_id=id, lease=None)
            return True, "Success", rid
        except Exception as e:
            return False, str(e), None

    def power_on(self):
        """Enble the motor power if e-stop is enabled."""
        try:
            power.power_on(self._power_client)
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def set_mobility_params(self, mobility_params):
        """Set Params for mobility and movement

        Args:
            mobility_params: spot.MobilityParams, params for spot mobility commands.
        """
        self._mobility_params = mobility_params

    def get_mobility_params(self):
        """Get mobility params
        """
        return self._mobility_params

    def arm_move_once_cmd(self, angles): # written by undergrads summer 2023 for spot arm visualization
       # import pdb
       # pdb.set_trace()
       # print(angles)
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(angles[0], angles[1], angles[2],
                angles[3], angles[4], angles[5]) # create trajectory destination from the given joint angles
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        command = make_robot_command(arm_joint_traj)
        cmd_id = self._robot_command(command) # make command to spot to move the joints 
       # print(self._robot_state_client.get_robot_state()) # for debugging; get the current robot state and print out the current joint angles

    def arm_pose_cmd(self, x, y, z, qx, qy, qz, qw, seconds=5):
        start = time.time()
        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self._robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, "hand")

        #dhand is desired hand pose
        odom_T_dhand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

        hand_pose_np = np.array([odom_T_hand.position.x, odom_T_hand.position.y, odom_T_hand.position.z])
        dhand_pose_np = np.array([odom_T_dhand.position.x, odom_T_dhand.position.y, odom_T_dhand.position.z])

        pos_diff_norm = np.linalg.norm(hand_pose_np-dhand_pose_np)

        threshold = 0.01

        if pos_diff_norm < threshold:
            # for small movements, increase duration of action - decreasing velocity?
            print("smol: ", pos_diff_norm)
            #return

        # duration in seconds is stored in seconds
        # arm_command = RobotCommandBuilder.arm_pose_command(
        #     odom_T_dhand.x, odom_T_dhand.y, odom_T_dhand.z, odom_T_dhand.rot.w, odom_T_dhand.rot.x,
        #     odom_T_dhand.rot.y, odom_T_dhand.rot.z, ODOM_FRAME_NAME, seconds)
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_dhand.x, odom_T_dhand.y, odom_T_dhand.z, odom_T_dhand.rot.w, odom_T_dhand.rot.x,
            odom_T_dhand.rot.y, odom_T_dhand.rot.z, ODOM_FRAME_NAME, seconds)
        #print(f'odom x: {odom_T_dhand.x}')
        #arm_command = RobotCommandBuilder.arm_gaze_command(
        #        odom_T_dhand.x, odom_T_dhand.y, odom_T_dhand.z, "trajectory?")
        
        #gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        #synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
        
        #gaze_command_id = self._robot_state_client.robot_command(synchro_command)

        #block_until_arm_arrives(command_client, gaze_command_id, 5.0)
        
        #return gaze_command_id
        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command((robot_state.manipulator_state.gripper_open_percentage)/100.0)
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send the request
        _, _, cmd_id = self._robot_command(arm_command)
        # cmd_id = self._robot_command(arm_command)
        
        print("Send follow command")
        success = block_until_arm_arrives(self._robot_command_client, cmd_id, 6.0)
        print(f"success: {success}")

        # if success == False:
        #     print("Move body")
        #     follow_arm_command = RobotCommandBuilder.follow_arm_command()
        #     command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)
        #     _, _, cmd_id = self._robot_command(follow_arm_command)

        #time.sleep(0.1)
        #return self._robot_command_client._get_robot_command_feedback_request(cmd_id)#WRITTEN BY THE UNDERGRADS (6/15/23 2pm)
        return self._robot_command_client.robot_command_feedback(cmd_id) # Are added this so we can access trajectory plan from feedback response
        # return cmd_id # Are added this so we can access trajectory plan from feSedback response

        #  rostopic pub -r 10 /arm_pose_stamped geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'


    def set_gripper(self, gripper_value):
        # Either open the gripper all the way, or slowly subtract from its value
        robot_state = self._robot_state_client.get_robot_state()

        if gripper_value > 0.0:
            # sets grip open to the passed in value
            robot_state.manipulator_state.gripper_open_percentage = gripper_value
        elif robot_state.manipulator_state.gripper_open_percentage >= -1.0 * gripper_value:
            # subtracts the amount of the command
            #robot_state.manipulator_state.gripper_open_percentage += gripper_value
            robot_state.manipulator_state.gripper_open_percentage = 0.0
        else:
            # set all the way to 0
            robot_state.manipulator_state.gripper_open_percentage = 0.0
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command((robot_state.manipulator_state.gripper_open_percentage)/100.0)
        cmd_id = self._robot_command(gripper_command)

        # Opens the gripper
        # rostopic pub -r 10 /spot/set_gripper geometry_msgs/Twist -- '[100.0, 0.0, 0.0]' '[0.0, 0.0, 0.0]'
    

    def arm_move_command(self, dx, dy, dz, dqx, dqy, dqz, dqw, cmd_duration=0.1):

        # get the current pose information
        robot_state = self._robot_state_client.get_robot_state()
        grav_aligned_body_T_hand = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
        print("The current arm position is {}".format(grav_aligned_body_T_hand))

        end_time=time.time() + cmd_duration

        # get_a_tform_b returns a SE3Pose, which stores x,y,z, and a quaternion rot
        # Calculate the new position
        x = grav_aligned_body_T_hand.x + dx
        y = grav_aligned_body_T_hand.y + dy
        z = grav_aligned_body_T_hand.z + dz

        qx = grav_aligned_body_T_hand.rot.x + dqx
        qy = grav_aligned_body_T_hand.rot.y + dqy
        qz = grav_aligned_body_T_hand.rot.z + dqz
        qw = grav_aligned_body_T_hand.rot.w + dqw

        #joystick
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        # call arm_pose_command
        self.arm_pose_cmd(x, y, z, qx, qy, qz, qw, seconds=cmd_duration)
        self._last_arm_command_time = end_time

        # Move forward
        # rostopic pub -r 100 /cmd_vel geometry_msgs/Twist -- '[0.5, 0.0, 0.0]' '[0.0, 0.0, 0.0]'

        # Move up
        # rostopic pub -r 100 /cmd_vel geometry_msgs/Twist -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, 0.0]'

        # Move right
        # rostopic pub -r 100 /cmd_vel geometry_msgs/Twist -- '[0.0, -0.5, 0.0]' '[0.0, 0.0, 0.0]'

        # Rotate base
        # rostopic pub -r 100 /cmd_vel geometry_msgs/Twist -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, 1.0]'

        # Move arm up
        # rostopic pub -r 100 /spot/arm_move geometry_msgs/Twist -- '[0.0, 0.3, 0.0]' '[0.0, 0.0, 0.0]'


    def velocity_arm_move_command(self, v_r=0.0, v_theta=0.0, v_z=0.0, v_rx=0.0, v_ry=0.0, v_rz=0.0, cmd_duration=0.1):

        # cylindrical_velocity
        # v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        # v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        # v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        # v_rx: angular velocity about X-axis in units rad/sec
        # v_ry: angular velocity about Y-axis in units rad/sec
        # v_rz: angular velocity about Z-axis in units rad/sec

        
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z 

        # angular velocity for gripper
        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        end_time=time.time() + cmd_duration
        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(end_time))

        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
            arm_velocity_command)

        # send command
        self._robot_command(robot_command, end_time_secs=end_time)

                                                                           

    def get_intrinsics(self):
        image_sources = self._image_client.list_image_sources()
        used_sources = ['hand_depth_in_hand_color_frame', 
                        'frontleft_depth_in_visual_frame', 
                        'frontright_depth_in_visual_frame', 
                        'left_depth_in_visual_frame',
                        'right_depth_in_visual_frame',
                        'back_depth_in_visual_frame']
        out_list = [0.0 for i in range(len(used_sources)*4)]
        for source in image_sources:
            if source.name in used_sources:
                start_ind = used_sources.index(source.name) * 4
                intrinsics = source.pinhole.intrinsics
                # FX
                out_list[start_ind] = intrinsics.focal_length.x
                # FY
                out_list[start_ind+1] = intrinsics.focal_length.y
                # CX
                out_list[start_ind+2] = intrinsics.principal_point.x
                # CY
                out_list[start_ind+3] = intrinsics.principal_point.y
        out_list = [float(x) for x in out_list]
        return out_list

    def arm_stow_command(self):
        #unstow = RobotCommandBuilder.arm_ready_command()
        stow = RobotCommandBuilder.arm_stow_command()

        carriable_and_stowable_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
            override_request=robot_state_proto.ManipulatorState.CARRY_STATE_CARRIABLE_AND_STOWABLE)
        grasp_holding_override = manipulation_api_pb2.ApiGraspOverride(
            override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING)

        override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
            api_grasp_override=grasp_holding_override,
            carry_state_override=carriable_and_stowable_override)
            

        self._manipulation_client.grasp_override_command(override_request)

        # Send the request
        stow_command_id = self._robot_command(stow)
        
        print("Stow command issued")
        # rostopic pub -r 10 /spot/arm_stow std_msgs/Bool -- 'True'

    def arm_unstow_command(self):
        #unstow = RobotCommandBuilder.arm_ready_command()
        unstow = RobotCommandBuilder.arm_ready_command()
        # Send the request
        unstow_command_id = self._robot_command(unstow)
        print("Unstow command issued")

        # rostopic pub -l /spot/arm_unstow geometry_msgs/Twist -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, 0.0]'


    # not in use currently. Gets publishable float 32 arrays based on the provided color and depth sources
    def getDepthData32(self, color_data, depth_data, color=False):        
        # Depth is a raw bytestream
        # import pdb
        cv_depth = np.frombuffer(depth_data.shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(depth_data.shot.image.rows,
                                    depth_data.shot.image.cols)

        #cv_depth is in millimeters, divide by 1000 to get it into meters
        cv_depth_meters = cv_depth / 1000.0

        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(color_data.shot.image.data, dtype=np.uint8), -1)
        
        
        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)
        visual_rgb = visual_rgb.astype('float') / 255.0

        return Float32MultiArray(None, cv_depth_meters.astype('float').flatten().tolist()), Float32MultiArray(None, visual_rgb.astype('float').flatten().tolist())


    def getDepthData8(self, image_responses):
        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
        # cv_depth is in millimeters, divide by 1000 to get it into meters
        # cv_depth_meters = cv_depth / 1000.0
        return cv_depth


    def velocity_cmd(self, v_x, v_y, v_rot, height, cmd_duration=0.1):
        """Send a velocity motion command to the robot.

        Args:
            v_x: Velocity in the X direction in meters
            v_y: Velocity in the Y direction in meters
            v_rot: Angular velocity around the Z axis in radians
            cmd_duration: (optional) Time-to-live for the command in seconds.  Default is 125ms (assuming 10Hz command rate).
        """
        # Are adding code to allow for changing spot body orientation, to raise the orientation of the front body cams
        footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=0.0) # Around 30degrees pitch rotation change pitch to -0.5
        # End of Are's added code
        end_time=time.time() + cmd_duration

        self._mobility_params = RobotCommandBuilder.mobility_params(body_height=height)#, footprint_R_body=footprint_R_body) # Added footprint_R_body to parameter - Are

        # note: (janeth) change command to include both arm and body movement
        # command = robot_command_pb2.RobotCommand()
        # # set the frame for the hand trajectory
        # # populate this command with points for the arm using 
        # command.synchronized_command.arm_command.arm_cartesian_command.root_frame_name = BODY_FRAME_NAME
        # point = command.synchronized_command.arm_command.arm_cartesian_command.pose_trajectory_in_task.points.add()

        # k = RobotCommandBuilder.
        

        # # set frame for body trajectory
        # command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = BODY_FRAME_NAME
        # END

        #### Calvin
        # Build the linear velocity command specified in a cylindrical coordinate system
        # cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        # cylindrical_velocity.linear_velocity.r = v_x
        # cylindrical_velocity.linear_velocity.theta = v_rot
        # cylindrical_velocity.linear_velocity.z = 0

        # arm_velocity_command = arm_command_pb2.ArmCommand.Request(
        #     cylindrical_velocity=cylindrical_velocity,
        #     end_time=self._robot.time_sync.robot_timestamp_from_local_secs(end_time))


        # # linear = geometry_pb2.Vec2(x=v_x, y=v_y)
        # # vel = geometry_pb2.SE2Velocity(linear=linear, angular=v_rot)
        # # mobility_command = mobility_command_pb2.MobilityCommand.Request(
        # #     se2_velocity_request=vel_command, params=self._mobility_params)

        # base_velocity_command = RobotCommandBuilder.synchro_velocity_command(
        #     v_x=v_x, 
        #     v_y=v_y,
        #     v_rot=v_rot, 
        #     params=self._mobility_params, 
        #     build_on_command=arm_velocity_command)

        # response = self._robot_command(base_velocity_command, end_time_secs=end_time, timesync_endpoint=self._robot.time_sync.endpoint)

        # arm_velocity_command = arm_command_pb2.ArmCommand.Request(
        #     cylindrical_velocity=cylindrical_velocity,
        #     end_time=self._robot.time_sync.robot_timestamp_from_local_secs(end_time))



        # command = robot_command_pb2.RobotCommand()
        # command.synchronized_command.mobility_command.se2_velocity_request.velocity.linear.x = v_x
        # command.synchronized_command.mobility_command.se2_velocity_request.velocity.linear.y = v_y
        # command.synchronized_command.mobility_command.se2_velocity_request.velocity.angular = v_rot
        # command.synchronized_command.mobility_command.se2_velocity_request.se2_frame_name = ODOM_FRAME_NAME

        # response = self._robot_command(command, end_time_secs=end_time, timesync_endpoint=self._robot.time_sync.endpoint)

        # command = synchro_velocity_command_stub(v_x=v_x, v_y=v_y, v_rot=v_rot, params=self._mobility_params)
        # response = self._robot_command(command)
        #### End Calvin

        response = self._robot_command(RobotCommandBuilder.synchro_velocity_command(
                                      v_x=v_x, v_y=v_y, v_rot=v_rot, params=self._mobility_params),
                                      end_time_secs=end_time, timesync_endpoint=self._robot.time_sync.endpoint)
                                      
        self._last_velocity_command_time = end_time
        # print("Velocity command response")
        # print(response)
        return response[0], response[1]

    def move_robot_using_ik(self, data):
        # rostopic pub -r 100 /spot/inv_kinematics geometry_msgs/Twist -- '[0.5, 0.0, 0.0]' '[0.0, 0.0, 0.0]'
        # rostopic pub -r 10 /spot/arm_stow std_msgs/Bool -- 'True'
        # bosdyn.client.robot.UnregisteredServiceNameError: service name "inverse-kinematics" has not been registered

        # CODE FOUND IN: /spot-sdk/python/examples/inverse_kinematics/reachability.py
        assert self._robot.has_arm(), 'Robot requires an arm to run IK'
        
        x_tomove = 0.5
        y_tomove = 0.5
        z_tomove = 0.5
        task_T_desired_tool = math_helpers.SE3Pose(x=x_tomove, y=y_tomove, z=0.0, rot=math_helpers.Quat())

        # These arrays store the reachability results as determined by the IK responses (`reachable_ik`)
        # or by trying to move to the desired tool pose (`reachable_cmd`).
        reachable_ik = False
        reachable_cmd = False

        # This list will store the (x, y) coordinates of the feet relative to the task frame, so that
        # the support polygon can be drawn in the output plot for reference.
        # foot_coords = []

        # Define a stand command that we'll send if the IK service does not find a solution.
        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.
            BodyAssistForManipulation(enable_hip_height_assist=True, enable_body_yaw_assist=True))
        body_assist_enabled_stand_command = RobotCommandBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control))

        # # Define the task frame to be in front of the robot and near the ground
        robot_state = self._robot_state_client.get_robot_state()
        odom_T_grav_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_gpe = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME,
                                   GROUND_PLANE_FRAME_NAME)

        # Construct the frame on the ground right underneath the center of the body.
        odom_T_ground_body = odom_T_grav_body
        odom_T_ground_body.z = odom_T_gpe.z

        # Now, construct a task frame slightly above the ground, in front of the robot.
        odom_T_task = odom_T_ground_body * math_helpers.SE3Pose(x=0.4, y=0, z=0.05, rot=math_helpers.Quat(w=1, x=0, y=0, z=0))

        # Now, let's set our tool frame to be the tip of the robot's bottom jaw. Flip the
        # orientation so that when the hand is pointed downwards, the tool's z-axis is
        # pointed upward.
        wr1_T_tool = math_helpers.SE3Pose(0.23589, 0, -0.03943, math_helpers.Quat.from_pitch(-math.pi / 2))

        # Populate the foot positions relative to the task frame for plotting later.
        odom_T_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                    ODOM_FRAME_NAME, BODY_FRAME_NAME)
        task_T_body = odom_T_task.inverse() * odom_T_body
        # for foot_index in [0, 1, 3, 2]:
        #     foot_state = robot_state.foot_state[foot_index]
        #     foot_position_rt_task = task_T_body.transform_vec3(foot_state.foot_position_rt_body)
        #     foot_coords.append((foot_position_rt_task.x, foot_position_rt_task.y))

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command(
            build_on_command=body_assist_enabled_stand_command)
        ready_command_id = self._robot_command(unstow)
        self._robot.logger.info('Going to "ready" pose')
        # print("READY ID: ", ready_command_id)
        block_until_arm_arrives(self._robot_command_client, ready_command_id[-1], 3.0) # function expects an integer for second index, but ready_command is a 3-element tuple

        # Create a client for the IK service.
        self._robot.sync_with_directory()
        ik_client = self._robot.ensure_client(InverseKinematicsClient.default_service_name)

        ik_request = InverseKinematicsRequest(
            root_frame_name=ODOM_FRAME_NAME,
            scene_tform_task=odom_T_task.to_proto(),
            wrist_mounted_tool=InverseKinematicsRequest.WristMountedTool(
                wrist_tform_tool=wr1_T_tool.to_proto()),
            tool_pose_task=InverseKinematicsRequest.ToolPoseTask(
                task_tform_desired_tool=task_T_desired_tool.to_proto()),
        )
        ik_responses = ik_client.inverse_kinematics(ik_request)
        reachable_ik = (ik_responses.status == InverseKinematicsResponse.STATUS_OK)

        # Attempt to move to each of the desired tool pose to check the IK results.
        stand_command = None
        if ik_responses.status == InverseKinematicsResponse.STATUS_OK:
            odom_T_desired_body = get_a_tform_b(
                ik_responses.robot_configuration.transforms_snapshot, ODOM_FRAME_NAME,
                BODY_FRAME_NAME)
            mobility_params = spot_command_pb2.MobilityParams(
                body_control=spot_command_pb2.BodyControlParams(
                    body_pose=RobotCommandBuilder.body_pose(ODOM_FRAME_NAME,
                                                            odom_T_desired_body.to_proto())))
            stand_command = RobotCommandBuilder.synchro_stand_command(params=mobility_params)
        else:
            stand_command = body_assist_enabled_stand_command
        arm_command = RobotCommandBuilder.arm_pose_command_from_pose(
            (odom_T_task * task_T_desired_tool).to_proto(), ODOM_FRAME_NAME, 1,
            build_on_command=stand_command)
        arm_command.synchronized_command.arm_command.arm_cartesian_command.wrist_tform_tool.CopyFrom(
            wr1_T_tool.to_proto())
        arm_command_id = self._robot_command(arm_command)
        print("arm command: ", arm_command_id)
        reachable_cmd = block_until_arm_arrives(self._robot_command_client, arm_command_id[-1], 2) # function expects an integer for second index, but ready_command is a 3-element tuple
        

        '''
        # The desired tool poses are defined relative to a task frame in front of the robot and slightly
        # above the ground. The task frame is aligned with the "gravity aligned body frame", such that
        # the positive-x direction is to the front of the robot, the positive-y direction is to the left
        # of the robot, and the positive-z direction is opposite to gravity.
        rng = np.random.RandomState(0)
        num_poses = 50
        x_size = 0.7  # m
        y_size = 0.8  # m
        x_rt_task = x_size * rng.random(num_poses)
        y_rt_task = -y_size / 2 + y_size * rng.random(num_poses)
        # z_rt_task = -z_size / 2 + z_size * rng.random(num_poses) # z position for future use
        # task_T_desired_tools = [
        #     math_helpers.SE3Pose(x=xi_rt_task, y=yi_rt_task, z=0.0, rot=math_helpers.Quat())
        #     for (xi_rt_task, yi_rt_task) in zip(x_rt_task.flatten(), y_rt_task.flatten())
        # ]

        # These arrays store the reachability results as determined by the IK responses (`reachable_ik`)
        # or by trying to move to the desired tool pose (`reachable_cmd`).
        reachable_ik = np.full(x_rt_task.shape, False)
        reachable_cmd = np.full(x_rt_task.shape, False)

        # This list will store the (x, y) coordinates of the feet relative to the task frame, so that
        # the support polygon can be drawn in the output plot for reference.
        foot_coords = []

        # Define a stand command that we'll send if the IK service does not find a solution.
        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.
            BodyAssistForManipulation(enable_hip_height_assist=True, enable_body_yaw_assist=True))
        body_assist_enabled_stand_command = RobotCommandBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control))

        # # Define the task frame to be in front of the robot and near the ground
        # robot_state = self._robot_state_client.get_robot_state()
        # odom_T_grav_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
        #                                  ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # odom_T_gpe = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME,
        #                            GROUND_PLANE_FRAME_NAME)

        # # Construct the frame on the ground right underneath the center of the body.
        # odom_T_ground_body = odom_T_grav_body
        # odom_T_ground_body.z = odom_T_gpe.z

        # Now, construct a task frame slightly above the ground, in front of the robot.
        # odom_T_task = odom_T_ground_body * math_helpers.SE3Pose(x=0.4, y=0, z=0.05, rot=math_helpers.Quat(w=1, x=0, y=0, z=0))

        # Now, let's set our tool frame to be the tip of the robot's bottom jaw. Flip the
        # orientation so that when the hand is pointed downwards, the tool's z-axis is
        # pointed upward.
        wr1_T_tool = math_helpers.SE3Pose(0.23589, 0, -0.03943, math_helpers.Quat.from_pitch(-math.pi / 2))

        # Populate the foot positions relative to the task frame for plotting later.
        # odom_T_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                    # ODOM_FRAME_NAME, BODY_FRAME_NAME)
        # # task_T_body = odom_T_task.inverse() * odom_T_body
        # for foot_index in [0, 1, 3, 2]:
        #     foot_state = robot_state.foot_state[foot_index]
        #     foot_position_rt_task = task_T_body.transform_vec3(foot_state.foot_position_rt_body)
        #     foot_coords.append((foot_position_rt_task.x, foot_position_rt_task.y))

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command(
            build_on_command=body_assist_enabled_stand_command)
        ready_command_id = self._robot_command(unstow)
        self._robot.logger.info('Going to "ready" pose')
        # print("READY ID: ", ready_command_id)
        block_until_arm_arrives(self._robot_command_client, ready_command_id[-1], 3.0) # function expects an integer for second index, but ready_command is a 3-element tuple

        # Create a client for the IK service.
        self._robot.sync_with_directory()
        ik_client = self._robot.ensure_client(InverseKinematicsClient.default_service_name)
        ik_responses = []
        for i, task_T_desired_tool in enumerate(task_T_desired_tools):
            # Query the IK service for the reachability of the desired tool pose.
            # Construct the IK request for this reachability problem. Note that since
            # `root_tform_scene` is unset, the "scene" frame is the same as the "root" frame in this
            # case.
            ik_request = InverseKinematicsRequest(
                root_frame_name=ODOM_FRAME_NAME,
                # scene_tform_task=odom_T_task.to_proto(),
                wrist_mounted_tool=InverseKinematicsRequest.WristMountedTool(
                    wrist_tform_tool=wr1_T_tool.to_proto()),
                tool_pose_task=InverseKinematicsRequest.ToolPoseTask(
                    task_tform_desired_tool=task_T_desired_tool.to_proto()),
            )
            ik_responses.append(ik_client.inverse_kinematics(ik_request))
            reachable_ik[i] = (ik_responses[i].status == InverseKinematicsResponse.STATUS_OK)

            # Attempt to move to each of the desired tool pose to check the IK results.
            stand_command = None
            if ik_responses[i].status == InverseKinematicsResponse.STATUS_OK:
                odom_T_desired_body = get_a_tform_b(
                    ik_responses[i].robot_configuration.transforms_snapshot, ODOM_FRAME_NAME,
                    BODY_FRAME_NAME)
                mobility_params = spot_command_pb2.MobilityParams(
                    body_control=spot_command_pb2.BodyControlParams(
                        body_pose=RobotCommandBuilder.body_pose(ODOM_FRAME_NAME,
                                                                odom_T_desired_body.to_proto())))
                stand_command = RobotCommandBuilder.synchro_stand_command(params=mobility_params)
            else:
                stand_command = body_assist_enabled_stand_command
            arm_command = RobotCommandBuilder.arm_pose_command_from_pose(
                (odom_T_task * task_T_desired_tool).to_proto(), ODOM_FRAME_NAME, 1,
                build_on_command=stand_command)
            arm_command.synchronized_command.arm_command.arm_cartesian_command.wrist_tform_tool.CopyFrom(
                wr1_T_tool.to_proto())
            arm_command_id = self._robot_command(arm_command)
            print("arm command: ", arm_command_id)
            reachable_cmd[i] = block_until_arm_arrives(self._robot_command_client, arm_command_id[-1], 2) # function expects an integer for second index, but ready_command is a 3-element tuple
            # import pdb
            # pdb.set_trace()

            '''
        

    def trajectory_cmd(self, goal_x, goal_y, goal_heading, cmd_duration, frame_name='odom', precise_position=False):
        """Send a trajectory motion command to the robot.

        Args:
            goal_x: Position X coordinate in meters
            goal_y: Position Y coordinate in meters
            goal_heading: Pose heading in radians
            cmd_duration: Time-to-live for the command in seconds.
            frame_name: frame_name to be used to calc the target position. 'odom' or 'vision'
            precise_position: if set to false, the status STATUS_NEAR_GOAL and STATUS_AT_GOAL will be equivalent. If
            true, the robot must complete its final positioning before it will be considered to have successfully
            reached the goal.
        """
        self._at_goal = False
        self._near_goal = False
        self._last_trajectory_command_precise = precise_position
        self._logger.info("got command duration of {}".format(cmd_duration))
        end_time=time.time() + cmd_duration
        if frame_name == 'vision':
            vision_tform_body = frame_helpers.get_vision_tform_body(
                    self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            body_tform_goal = math_helpers.SE3Pose(x=goal_x, y=goal_y, z=0, rot=math_helpers.Quat.from_yaw(goal_heading))
            vision_tform_goal = vision_tform_body * body_tform_goal
            response = self._robot_command(
                            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                goal_x=vision_tform_goal.x,
                                goal_y=vision_tform_goal.y,
                                goal_heading=vision_tform_goal.rot.to_yaw(),
                                frame_name=frame_helpers.VISION_FRAME_NAME,
                                params=self._mobility_params),
                            end_time_secs=end_time
                            )
        elif frame_name == 'odom':
            odom_tform_body = frame_helpers.get_odom_tform_body(
                    self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            body_tform_goal = math_helpers.SE3Pose(x=goal_x, y=goal_y, z=0, rot=math_helpers.Quat.from_yaw(goal_heading))
            odom_tform_goal = odom_tform_body * body_tform_goal
            response = self._robot_command(
                            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                goal_x=odom_tform_goal.x,
                                goal_y=odom_tform_goal.y,
                                goal_heading=odom_tform_goal.rot.to_yaw(),
                                frame_name=frame_helpers.ODOM_FRAME_NAME,
                                params=self._mobility_params),
                            end_time_secs=end_time
                            )
        else:
            raise ValueError('frame_name must be \'vision\' or \'odom\'')
        if response[0]:
            self._last_trajectory_command = response[2]
        return response[0], response[1]

    def list_graph(self, upload_path):
        """List waypoint ids of garph_nav
        Args:
          upload_path : Path to the root directory of the map.
        """
        ids, eds = self._list_graph_waypoint_and_edge_ids()
        # skip waypoint_ for v2.2.1, skip waypiont for < v2.2
        return [v for k, v in sorted(ids.items(), key=lambda id : int(id[0].replace('waypoint_','')))]

    def navigate_to(self, upload_path,
                    navigate_to,
                    initial_localization_fiducial=True,
                    initial_localization_waypoint=None):
        """ navigate with graph nav.

        Args:
           upload_path : Path to the root directory of the map.
           navigate_to : Waypont id string for where to goal
           initial_localization_fiducial : Tells the initializer whether to use fiducials
           initial_localization_waypoint : Waypoint id string of current robot position (optional)
        """
        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            upload_filepath = upload_path[:-1]
        else:
            upload_filepath = upload_path

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # FIX ME somehow,,,, if the robot is stand, need to sit the robot before starting garph nav
        if self.is_standing and not self.is_moving:
            self.sit()

        # TODO verify estop  / claim / power_on
        self._clear_graph()
        self._upload_graph_and_snapshots(upload_filepath)
        if initial_localization_fiducial:
            self._set_initial_localization_fiducial()
        if initial_localization_waypoint:
            self._set_initial_localization_waypoint([initial_localization_waypoint])
        self._list_graph_waypoint_and_edge_ids()
        self._get_localization_state()
        resp = self._navigate_to([navigate_to])

        return resp

    ## copy from spot-sdk/python/examples/graph_nav_command_line/graph_nav_command_line.py
    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        self._logger.info('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        self._logger.info('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            self._logger.error("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance = 0.2,
            max_yaw = 20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            self._logger.error("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id, self._logger)
        return self._current_annotation_name_to_wp_id, self._current_edges


    def _upload_graph_and_snapshots(self, upload_filepath):
        """Upload the graph and snapshots to the robot."""
        self._logger.info("Loading the graph from disk into local storage...")
        with open(upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            self._logger.info("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            # Load the edge snapshots from disk.
            with open(upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        self._logger.info("Uploading the graph and snapshots to the robot...")
        self._graph_nav_client.upload_graph(lease=self._lease.lease_proto,
                                            graph=self._current_graph)
        # Upload the snapshots to the robot.
        for waypoint_snapshot in self._current_waypoint_snapshots.values():
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            self._logger.info("Uploaded {}".format(waypoint_snapshot.id))
        for edge_snapshot in self._current_edge_snapshots.values():
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            self._logger.info("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and it if is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            self._logger.info(
                   "Upload complete! The robot is currently not localized to the map; please localize", \
                   "the robot using commands (2) or (3) before attempting a navigation command.")

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            self._logger.info("No waypoint provided as a destination for navigate to.")
            return

        self._lease = self._lease_wallet.get_lease()
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            self._logger.info("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keepalive and create a new sublease for graph nav.
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()

        # Navigate to the destination waypoint.
        is_finished = False
        nav_to_cmd_id = -1
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                               leases=[sublease.lease_proto])
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Update the lease and power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

        status = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            return True, "Successfully completed the navigation commands!"
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            return False, "Robot got lost when navigating the route, the robot will now sit down."
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            return False, "Robot got stuck when navigating the route, the robot will now sit down."
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            return False, "Robot is impaired."
        else:
            return False, "Navigation command is not complete yet."

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            self._logger.error("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                self._logger.error("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                self._logger.error(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge."
                )
                break

        self._lease = self._lease_wallet.get_lease()
        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                self._logger.error("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Stop the lease keepalive and create a new sublease for graph nav.
            self._lease = self._lease_wallet.advance()
            sublease = self._lease.create_sublease()
            self._lease_keepalive.shutdown()

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0, leases=[sublease.lease_proto])
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            self._lease = self._lease_wallet.advance()
            self._lease_keepalive = LeaseKeepAlive(self._lease_client)

            # Update the lease and power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph(lease=self._lease.lease_proto)

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(timeout=10) # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have not status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            self._logger.error("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            self._logger.error("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            self._logger.error("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None
