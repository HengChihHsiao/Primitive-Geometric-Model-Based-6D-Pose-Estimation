import rospy
import ros_numpy
import moveit_commander
import tf
import numpy as np
import open3d as o3d
import copy
import math
import random
import sys
import cv2
import time

from typing import Any
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial.transform import Rotation
from yolov8_ros_msgs.msg import BoundingBox, Instance_seg, Instance_segs

yolo_msgs = None
yolo_result_frame = None

rs_cloud = None
rs_depth_frame = None
rs_color_frame = None
intrinstic_matrix = np.array([[1043.99267578125, 0.0, 960.0],
                              [0.0, 1043.99267578125,  540.0],
                              [0.0, 0.0, 1.0]])

def yolo_callback(msg):
    global yolo_msgs
    yolo_msgs = msg
    return yolo_msgs


def yolo_result_callback(color_msg):
    global yolo_result_frame
    assert isinstance(color_msg, Image)
    yolo_result_frame = CvBridge().imgmsg_to_cv2(color_msg, 'bgr8')
    return yolo_result_frame


def rs_color_callback(color_msg):
    global rs_color_frame
    assert isinstance(color_msg, Image)
    rs_color_frame = CvBridge().imgmsg_to_cv2(color_msg, 'bgr8')
    return rs_color_frame


def rs_depth_callback(depth_msg):
    global rs_depth_frame
    assert isinstance(depth_msg, Image)
    rs_depth_frame = CvBridge().imgmsg_to_cv2(depth_msg, '16UC1')
    return rs_depth_frame


def rs_points_callback(points_msg):
    # rospy.sleep(3)
    global rs_cloud
    assert isinstance(points_msg, PointCloud2)
    pc = ros_numpy.numpify(points_msg)
    points = np.zeros((pc.shape[0], 3))
    points[:, 0] = pc['x']
    points[:, 1] = pc['y']
    points[:, 2] = pc['z']
    rs_cloud = o3d.geometry.PointCloud()
    rs_cloud.points = o3d.utility.Vector3dVector(points)
    # print('cb', cloud)

    return rs_cloud

def get_rotation_matrix(quaternion):
    '''
    Get the rotation matrix from quaternion
    args:
        quaternion : list of float (x, y, z, w)
    return:
        ratation_matrix : numpy array
    '''
    # ratation_matrix = Rotation.from_quat(quaternion).as_matrix()
    x, y, z, w = quaternion
    ratation_matrix = [[1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]]

    return ratation_matrix

def get_CameraFrame_pos(u, v, depth_value):
    '''
    Get the position of the point in the camera frame
    args:
        u : float (x-axis pixel)
        v : float (y-axis pixel)
        depth_value : float (depth)
    return:
        position : numpy array
    '''
    fx = intrinstic_matrix[0][0]
    fy = intrinstic_matrix[1][1]
    cx = intrinstic_matrix[0][2]
    cy = intrinstic_matrix[1][2]

    z = depth_value
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return [x, y, z, 1]

def get_RT_matrix(base_frame: Any,
                  reference_frame: Any):
    
    '''
    Get the transformation matrix from base frame to reference frame
    args:
        base_frame : str (base frame)
        reference_frame : str (reference frame)
    return:
        RT_matrix : list of [ndarray(R), ndarray(T)]
    '''
    listener = tf.TransformListener()
    i = 3
    while i != 0:
        try:
            listener.waitForTransform(base_frame, reference_frame, rospy.Time.now(), rospy.Duration(3.0))
            camera2World = listener.lookupTransform(base_frame, reference_frame, rospy.Time(0))
            T = camera2World[0]
            R = get_rotation_matrix(camera2World[1])
            R[0].append(0)
            R[1].append(0)
            R[2].append(0)
            R.append([0.0, 0.0, 0.0, 1.0])
            R = np.mat(R)
            return [R, T]
        except:
            rospy.loginfo('tf error!!!')
        i = i - 1

def coordinate_transform(CameraFrame_pos: np.ndarray,
                         R: np.ndarray,
                         T: np.ndarray) -> np.ndarray:
    '''
    Transform the position from camera frame to world frame
    args:
        CameraFrame_pos : numpy array
        R : numpy array
        T : numpy array
    return:
        WorldFrame_pos : numpy array
    '''
    WorldFrame_pos = R.I * np.mat(CameraFrame_pos).T
    WorldFrame_pos[0, 0] = WorldFrame_pos[0, 0] + T[0]
    WorldFrame_pos[1, 0] = WorldFrame_pos[1, 0] + T[1]
    WorldFrame_pos[2, 0] = WorldFrame_pos[2, 0] + T[2]
    WorldFrame_pos = [WorldFrame_pos[0, 0],
                      WorldFrame_pos[1, 0], 
                      WorldFrame_pos[2, 0]]
    return WorldFrame_pos

def camera_to_pixel_coordinate(camera_coordinate: np.ndarray) -> int:
    '''
    Get the pixel coordinate of the point in the image
    args:
        camera_coordinate : numpy array
    return:
        pixel_coordinate : numpy array
    '''
    fx = intrinstic_matrix[0][0]
    fy = intrinstic_matrix[1][1]
    cx = intrinstic_matrix[0][2]
    cy = intrinstic_matrix[1][2]

    x = camera_coordinate[0]
    y = camera_coordinate[1]
    z = camera_coordinate[2]

    u = x * fx / z + cx
    v = y * fy / z + cy

    return u, v

class Robotic_Arm:
    def __init__(self) -> None:
        moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
        
        self.reference_frame = 'world'
        self.robot = moveit_commander.robot.RobotCommander()
        self.arm = moveit_commander.move_group.MoveGroupCommander('arm_joint')
        self.end_effector_link = self.arm.get_end_effector_link()

        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.arm.allow_replanning(True)
        self.arm.set_pose_reference_frame(self.reference_frame)
    
    def get_current_pose(self) -> list:
        '''
        Get the current pose of the end effector
        return:
            pose : list of float (x, y, z, x, y, z, w)
        '''
        pose = self.arm.get_current_pose(self.end_effector_link)
        return pose

    @property
    def poseNow(self) -> list:
        '''
        Get the current pose of the end effector
        return:
            pose : list of float (x, y, z, x, y, z, w)
        '''
        return self.arm.get_current_pose(self.end_effector_link)
    
    def get_current_joint_angle(self) -> list:
        '''
        Get the current joint angle of the end effector
        return:
            joint_angle : list of float (joint1, joint2, joint3, joint4, joint5, joint6)
        '''
        joint_angle = self.arm.get_current_joint_values()
        return joint_angle
    
    def get_target_pose(self, position, orientation) -> list:
        '''
        Get the target pose of the end effector
        args:
            position : list of float (x, y, z)
            orientation : list of float (x, y, z, w)
        return:
            pose : list of float (x, y, z, x, y, z, w)
        '''

        target_pose = self.poseNow.pose
        target_pose.position.x = position[0]
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]
        target_pose.orientation.x = orientation[0]
        target_pose.orientation.y = orientation[1]
        target_pose.orientation.z = orientation[2]
        target_pose.orientation.w = orientation[3]

        return target_pose

    def move(self, 
             position : list, 
             orientation : list) -> bool:
        ''' 
        Move the end effector to the target position and orientation
        args:
            position : list of float (x, y, z)
            orientation : list of float (x, y, z, w)
        return:
            status : bool (True if success, False if fail)
        '''
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = position[0]
        target_pose.pose.position.y = position[1]
        target_pose.pose.position.z = position[2]
        target_pose.pose.orientation.x = orientation[0]
        target_pose.pose.orientation.y = orientation[1]
        target_pose.pose.orientation.z = orientation[2]
        target_pose.pose.orientation.w = orientation[3]
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        # print('pose', self.poseNow)
        status = self.arm.go(wait=True)
        # print('pose_after_go', self.poseNow)

        return status
    
    def move_with_angle(self, 
                        joint_angle : list) -> bool:
        '''
        Move the end effector to the target joint angle
        args:
            joint_angle : list of float (joint1, joint2, joint3, joint4, joint5, joint6)
        return:
            status : bool (True if success, False if fail)
        '''
        self.arm.set_joint_value_target(joint_angle)
        self.arm.set_start_state_to_current_state()
        status = self.arm.go(wait=True)
        return status
    
    def move_by_position(self, 
                         position : list) -> bool:
        '''
        Move the end effector by the target pose and orientation
        args:
            position : list of float (x, y, z)
        return:
            status : bool (True if success, False if fail)
        '''
        self.arm.set_pose_target(position, self.end_effector_link)
        self.arm.set_start_state_to_current_state()
        status = self.arm.go(wait=True)
        return status
    
    def move_cartesian_by_waypoints(self,
                                    waypoints : list) -> bool:
        '''
        Move the end effector by the target waypoints
        args:
            waypoints : list of list of waypoints (x, y, z, x, y, z, w)
        return:
            status : bool (True if success, False if fail)
        '''
        fraction = 0.0
        maxtries = 100
        attempts = 0
        self.arm.set_pose_reference_frame(self.reference_frame)
        self.arm.set_start_state_to_current_state()
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints, 0.01, 0.0, True)
            attempts += 1

            if fraction == 1.0:
                status = self.arm.execute(plan, wait=True)
                return status
            else:
                rospy.loginfo("Path planning failed with only {0:.4f} success after {1} attempts.".format(fraction, maxtries))
                return False
    
    def go_named(self, 
                      pose_name : str) -> bool:
        '''
        Move the end effector to the target pose
        args:
            pose_name : str (named pose)
        return:
            status : bool (True if success, False if fail)
        '''
        self.arm.set_named_target(pose_name)
        self.arm.set_start_state_to_current_state()
        status = self.arm.go(wait=True)
        return status
    

class Pose_estimation:
    def __init__(self) -> None:
        self.plane_seg_param_distance_threshold = 0.01
        self.plane_seg_param_ransac_n = 5
        self.plane_seg_param_num_iterations = 1000

        self.search_neighbor = False

        self.visual_front = np.array([-0.0009970323721936443, -0.0083030475574617547, -0.999965032052475])
        self.visual_lookat = np.array([0.059073498605307001, 0.02572381533381584, 0.99301894268958202])
        self.visual_up = np.array([0.0016727088200268305, -0.99996414398807609, 0.0083013723804005075])
        self.visual_zoom = 0.23999999999999957

        self.model_color = np.array([1, 0.706, 0])
        self.object_color = np.array([0, 0.651, 0.929])
        self.other_color = np.array([0, 0, 1])

        self.voxel_size = 0.008


    def pcd_plane_segmentation(self, 
                               pc : o3d.geometry.PointCloud,
                               distance_threshold : float = 0.01,
                               ransac_n : int = 5,
                               num_iterations : int = 1000) -> o3d.geometry.PointCloud:
        '''
        Plane segmentation of the point cloud
        args:
            pc : open3d.geometry.PointCloud
            distance_threshold : float (distance threshold for plane segmentation)
            ransac_n : int (number of points to sample for generating a plane)
            num_iterations : int (number of iterations for plane segmentation)
        return:
            inlier_cloud : open3d.geometry.PointCloud (inlier point cloud)
            outlier_cloud : open3d.geometry.PointCloud (outlier point cloud)
        '''
        distance_threshold = self.plane_seg_param_distance_threshold
        ransac_n = self.plane_seg_param_ransac_n
        num_iterations = self.plane_seg_param_num_iterations
        plane_model, inliers = pc.segment_plane(distance_threshold, 
                                                   ransac_n, 
                                                   num_iterations)
        [a, b, c, d] = plane_model
        inlier_cloud = pc.select_by_index(inliers)
        outlier_cloud = pc.select_by_index(inliers, invert=True)
        return inlier_cloud, outlier_cloud
    
    def pixel_to_camera_coordinate(self,
                                   depth_frame : np.ndarray,
                                   u : float,
                                   v : float,
                                   search_neighbor: bool = False) -> np.ndarray:
        '''
        Get the position of the point in the camera frame
        args:
            depth_frame : numpy array (depth frame)
            u : float (x-axis pixel)
            v : float (y-axis pixel)
            search_region : int (search region)
        return:
            camera_coordinate : numpy array
        '''
        fx_d = intrinstic_matrix[0][0]
        fy_d = intrinstic_matrix[1][1]
        cx_d = intrinstic_matrix[0][2]
        cy_d = intrinstic_matrix[1][2]

        xyz = np.zeros(3)
        Z = depth_frame[int(v), int(u)] / 1000.0
        if search_neighbor:
            u_temp, v_temp = u, v
            search_region = 1
            while Z == 0:
                for i in range(-search_region, search_region + 1):
                    for j in range(-search_region, search_region + 1):
                        Z = depth_frame[int(v_temp + i), int(u_temp + j)] / 1000.0
                        if Z != 0:
                            break
                    if Z != 0:
                        break
                search_region += 1
        
        X = (u - cx_d) * Z / fx_d
        Y = (v - cy_d) * Z / fy_d
        xyz = [X, Y, Z]
        return xyz

    def get_bbox_camera_coordinate(self,
                                   depth_frame : np.ndarray,
                                   bbox : tuple) -> tuple:
        '''
        Get the bounding box in the camera frame
        args:
            depth_frame : numpy array (depth frame)
            bbox : tuple (xmin, xmax, ymin, ymax)
        return:
            bbox_camera_coordinate : tuple (xmin, xmax, ymin, ymax)
        '''
        xmin, ymin, xmax, ymax = bbox
        search_neighbor = False
        world_bbox = np.array([
            self.pixel_to_camera_coordinate(depth_frame, xmin, ymin, search_neighbor),
            self.pixel_to_camera_coordinate(depth_frame, xmax, ymin, search_neighbor),
            self.pixel_to_camera_coordinate(depth_frame, xmax, ymax, search_neighbor),
            self.pixel_to_camera_coordinate(depth_frame, xmin, ymax, search_neighbor)
        ])

        return world_bbox

    def crop_pc_by_bbox(self,
                        depth_frame : np.ndarray,
                        pc: o3d.geometry.PointCloud,
                        bbox : tuple) -> o3d.geometry.PointCloud:
        '''
        Crop the point cloud by the bounding box
        args:
            depth_frame : numpy array (depth frame)
            pc : open3d.geometry.PointCloud
            bbox : tuple (xmin, xmax, ymin, ymax)
        return:
            crop_pc : open3d.geometry.PointCloud
        '''
        bbox_3d = self.get_bbox_camera_coordinate(depth_frame, bbox)

        # cropped_pc = pc.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_3d.min(axis=0),
                                                                # max_bound=bbox_3d.max(axis=0)))
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = 'Z'
        vol.axis_max = 2
        vol.axis_min = -2
        vol.bounding_polygon = o3d.utility.Vector3dVector(bbox_3d)
        cropped_pc = vol.crop_point_cloud(pc)

        return cropped_pc
    
    def draw_registration_result(self, 
                                 source: o3d.geometry.PointCloud,
                                 target: o3d.geometry.PointCloud,
                                 transformation: np.ndarray,
                                 windows_name: str = 'Registration result') -> None:
        '''
        Draw the registration result
        args:
            source : open3d.geometry.PointCloud
            target : open3d.geometry.PointCloud
            transformation : numpy array
            windows_name : str (windows name)
        '''
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color(self.model_color)
        # target_temp.paint_uniform_color(self.object_color)
        if transformation is not None:
            source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], 
                                          windows_name,
                                          front = self.visual_front, 
                                          lookat = self.visual_lookat, 
                                          up = self.visual_up, 
                                          zoom = self.visual_zoom)
        return None

    def preprocess_point_cloud(self,
                               pc: o3d.geometry.PointCloud,
                               voxel_size: float) -> o3d.geometry.PointCloud:
        '''
        Preprocess the point cloud
        args:
            pc : open3d.geometry.PointCloud
            voxel_size : float (voxel size)
        return:
            pcd_down : open3d.geometry.PointCloud
            pcd_fpth : open3d.geometry.PointCloud
        '''
        pcd_down = pc.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        return pcd_down, pcd_fpfh
    
    def prepare_dataset(self,
                        source_pc: o3d.geometry.PointCloud,
                        target_pc: o3d.geometry.PointCloud,
                        voxel_size: float) -> list:
        '''
        Prepare the dataset for registration
        args:
            source_pc : open3d.geometry.PointCloud
            target_pc : open3d.geometry.PointCloud
            voxel_size : float (voxel size)
        return:
            source_pc : open3d.geometry.PointCloud
            target_pc : open3d.geometry.PointCloud
            source_down : open3d.geometry.PointCloud
            target_down : open3d.geometry.PointCloud
            source_fpfh : open3d.geometry.PointCloud
            target_fpfh : open3d.geometry.PointCloud
        '''
        radius_normal = voxel_size * 2
        source_down, source_fpfh = self.preprocess_point_cloud(source_pc, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target_pc, voxel_size)
        target_pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        return source_pc, target_pc, source_down, target_down, source_fpfh, target_fpfh
    
    def execute_global_registration(self,
                                    source_down: o3d.geometry.PointCloud,
                                    target_down: o3d.geometry.PointCloud,
                                    source_fpfh: o3d.geometry.PointCloud,
                                    target_fpfh: o3d.geometry.PointCloud,
                                    voxel_size: float) -> Any:
        '''
        Execute the global registration
        args:
            source_down : open3d.geometry.PointCloud
            target_down : open3d.geometry.PointCloud
            source_fpfh : open3d.geometry.PointCloud
            target_fpfh : open3d.geometry.PointCloud
            voxel_size : float (voxel size)
        return:
            result : open3d.registration.RegistrationResult
        '''
        distance_threshold = voxel_size * 1.5
        RANSAC_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, 
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
        )
        return RANSAC_result
    
    def execute_fast_global_registration(self,
                                         source_down: o3d.geometry.PointCloud,
                                         target_down: o3d.geometry.PointCloud,
                                         source_fpfh: o3d.geometry.PointCloud,
                                         target_fpfh: o3d.geometry.PointCloud,
                                         voxel_size: float) -> Any:
        '''
        Execute the fast global registration
        args:
            source_down : open3d.geometry.PointCloud
            target_down : open3d.geometry.PointCloud
            source_fpfh : open3d.geometry.PointCloud
            target_fpfh : open3d.geometry.PointCloud
            voxel_size : float (voxel size)
        return:
            result : open3d.registration.RegistrationResult
        '''
        distance_threshold = voxel_size * 0.5
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            )
        )
        return result
    
    def refine_registration(self,
                            source_pc: o3d.geometry.PointCloud,
                            target_pc: o3d.geometry.PointCloud,
                            voxel_size: float,
                            RANSAC_result: Any) -> Any:
        '''
        Refine the registration
        args:
            source_pc : open3d.geometry.PointCloud
            target_pc : open3d.geometry.PointCloud
            voxel_size : float (voxel size)
            result : open3d.registration.RegistrationResult
        return:
            result : open3d.registration.RegistrationResult
        '''
        # distance_threshold = voxel_size * 0.4
        distance_threshold = 0.02
        ICP_result = o3d.pipelines.registration.registration_icp(
            source_pc,
            target_pc,
            distance_threshold,
            RANSAC_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
        )
        return ICP_result
    
    def create_vis_grasp_pc(self,
                            grasp_point_left_pc : o3d.geometry.PointCloud,
                            grasp_point_right_pc : o3d.geometry.PointCloud,
                            extend_param : int) -> o3d.geometry.PointCloud:
        '''
        Create the visualization point cloud of the grasp point to extend the grasp point in x-axis
        args:
            grasp_point_left_pc : open3d.geometry.PointCloud which only contains the left grasp point
            grasp_point_right_pc : open3d.geometry.PointCloud which only contains the right grasp point
            extend_param : int (the number of points to extend)
        return:
            vis_grasp_point_left_pc : open3d.geometry.PointCloud
            vis_grasp_point_right_pc : open3d.geometry.PointCloud
        '''
        vis_grasp_point_left_pc = copy.deepcopy(grasp_point_left_pc)
        vis_grasp_point_right_pc = copy.deepcopy(grasp_point_right_pc)
        vis_grasp_point_left = np.zeros((extend_param, 3))
        vis_grasp_point_right = np.zeros((extend_param, 3))
        grasp_point_left = np.asarray(grasp_point_left_pc.points[0])
        grasp_point_right = np.asarray(grasp_point_right_pc.points[0])
        for i in range(extend_param):
            vis_grasp_point_left[i] = [grasp_point_left[0] - i * 0.0001, grasp_point_left[1], grasp_point_left[2]]
            vis_grasp_point_right[i] = [grasp_point_right[0] + i * 0.0001, grasp_point_right[1], grasp_point_right[2]]       

        vis_grasp_point_left_pc.points = o3d.utility.Vector3dVector(vis_grasp_point_left)
        vis_grasp_point_right_pc.points = o3d.utility.Vector3dVector(vis_grasp_point_right)
        vis_grasp_point_left_pc.paint_uniform_color([1, 0, 0])
        vis_grasp_point_right_pc.paint_uniform_color([1, 0, 0])

        return vis_grasp_point_left_pc, vis_grasp_point_right_pc
    
    def cal_grasp_angle_from_two_grasp_points(self,
                                              grasp_point_left : np.ndarray,
                                              grasp_point_right : np.ndarray) -> float:
                                              
        grasp_v = np.array(grasp_point_left) - np.array(grasp_point_right)
        target_orientation = np.array([0, 1, 0])
        norm_rt = np.linalg.norm(target_orientation)
        norm_trans = np.linalg.norm(grasp_v)
        grasp_angle = np.dot(grasp_v, target_orientation) / (norm_trans * norm_rt)
        grasp_angle = np.arccos(grasp_angle) * 180 / np.pi

        # if grasp_angle > 90:
        #     grasp_angle = -(180 - grasp_angle)

        # if grasp_point_left[1] < grasp_point_right[1]:
        #     grasp_angle = -grasp_angle

        return grasp_angle
    
    def find_grasp_point(self,
                         rs_color_frame : np.ndarray,
                         rs_depth_frame : np.ndarray,
                         visualize: bool = False) -> tuple:
        '''
        Find the grasp point of the object
        args:
            rs_color_frame : numpy array (color frame)
            rs_depth_frame : numpy array (depth frame)
            visualize : bool (True if visualize, False if not)
        return:
            grasp_point_left : numpy array
            grasp_point_middle : numpy array
            grasp_point_right : numpy array
            est_success : bool (True if success, False if fail)
        '''

        object_dected_flag = False
        est_success = False
        class_name = ''
        height, width = rs_depth_frame.shape

        rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rs_color_frame),
            o3d.geometry.Image(rs_depth_frame),
        )
        whole_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_frame,
            o3d.camera.PinholeCameraIntrinsic(width, height, intrinstic_matrix)
        )

        time_start = time.time()
        inlier_pc, outlier_pc = self.pcd_plane_segmentation(whole_pc)
        plane_seg_time = time.time() - time_start

        if visualize:
            o3d.visualization.draw_geometries([whole_pc], 
                                              window_name='whole_pc', 
                                              front = self.visual_front, 
                                              lookat = self.visual_lookat, 
                                              up = self.visual_up, 
                                              zoom = self.visual_zoom)
            o3d.visualization.draw_geometries([copy.copy(inlier_pc).paint_uniform_color(self.other_color), outlier_pc], 
                                              window_name='plane_seg', 
                                              front = self.visual_front, 
                                              lookat = self.visual_lookat, 
                                              up = self.visual_up, 
                                              zoom = self.visual_zoom)
        
        for instance_msg in yolo_msgs.instance_segs:
            if instance_msg.boundingbox.Class == 'bottle' or instance_msg.boundingbox.Class == 'box' or instance_msg.boundingbox.Class == 'sphere':
                object_dected_flag = True
                class_name = instance_msg.boundingbox.Class

                x_min = instance_msg.boundingbox.xmin
                x_max = instance_msg.boundingbox.xmax
                y_min = instance_msg.boundingbox.ymin
                y_max = instance_msg.boundingbox.ymax
                object_bbox = [x_min, y_min, x_max, y_max]
            
            if object_dected_flag:
                time_start = time.time()
                object_pc = self.crop_pc_by_bbox(rs_depth_frame, outlier_pc, object_bbox)
                crop_object_pc_time = time.time() - time_start
                if visualize:
                    o3d.visualization.draw_geometries([copy.copy(whole_pc).paint_uniform_color(self.other_color), object_pc], 
                                                      window_name='object_pc', 
                                                      front = self.visual_front, 
                                                      lookat = self.visual_lookat, 
                                                      up = self.visual_up, 
                                                      zoom = self.visual_zoom)
                    o3d.visualization.draw_geometries([object_pc], window_name='object_pc')

                obb = object_pc.get_minimal_oriented_bounding_box(robust=True)
                obb.color = (1, 0, 0)
                [center_x, center_y, center_z] = obb.get_center()
                vertices = obb.get_box_points()
                vertices = np.asarray(vertices)
                # print("=======================================================")
                # print(vertices)
                # print("=======================================================")

                FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

                if visualize:
                    o3d.visualization.draw_geometries([object_pc, obb, FOR1], window_name='axis_aligned_bounding_box')

                ################ prepare registration dataset ################
                voxel_size = self.voxel_size
                object_scale = object_pc.get_max_bound() - object_pc.get_min_bound()
                object_center = object_pc.get_center()

                def cal_hypotenuse_3d(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
                    return math.sqrt(math.pow(vertices_a[0] - vertices_b[0], 2) + 
                                     math.pow(vertices_a[1] - vertices_b[1], 2) + 
                                     math.pow(vertices_a[2] - vertices_b[2], 2))
                
                if class_name == 'bottle':
                    object_radius = cal_hypotenuse_3d(vertices[0], vertices[2]) / 2
                    object_height = cal_hypotenuse_3d(vertices[3], vertices[4])

                    time_start = time.time()
                    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=object_radius, height=object_height)
                    model_pc = cylinder_mesh.sample_points_poisson_disk(number_of_points=10000)
                    model_generate_time = time.time() - time_start
                    model_rotate_R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, -np.pi, -np.pi))
                    model_pc = model_pc.rotate(model_rotate_R, center=model_pc.get_center())

                elif class_name == 'box':
                    object_width = cal_hypotenuse_3d(vertices[0], vertices[1])
                    object_depth = cal_hypotenuse_3d(vertices[0], vertices[3])
                    object_length = cal_hypotenuse_3d(vertices[3], vertices[5])

                    if object_width > object_length:
                        model_rotate_R = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, np.pi / 2))
                    else:
                        model_rotate_R = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, 0))
                    box_mesh = o3d.geometry.TriangleMesh.create_box(width=object_width, height=object_length, depth=object_depth)
                    
                    time_start = time.time()
                    model_pc = box_mesh.sample_points_poisson_disk(number_of_points=10000)
                    model_generate_time = time.time() - time_start
                    model_pc = model_pc.paint_uniform_color(self.model_color)
                    model_pc.translate([0, 0, 0], relative=False)
                    model_pc = model_pc.rotate(model_rotate_R, center=model_pc.get_center())

                elif class_name == 'sphere':
                    time_start = time.time()
                    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=object_scale[0] / 2)
                    model_pc = sphere_mesh.sample_points_poisson_disk(number_of_points=10000)
                    model_generate_time = time.time() - time_start

                model_pc_wo_voxel = copy.deepcopy(model_pc).paint_uniform_color(self.model_color)
                model_pc = model_pc.voxel_down_sample(voxel_size=voxel_size)
                model_pc = model_pc.translate(object_center, relative=False)


                if class_name == 'bottle' or class_name == 'sphere':
                    min_dist = model_pc_wo_voxel.compute_nearest_neighbor_distance()
                    candidate = np.where((np.asarray(model_pc_wo_voxel.points)[:, 1] - model_pc_wo_voxel.get_center()[1] < min_dist[0]) &
                                        (np.asarray(model_pc_wo_voxel.points)[:, 1] - model_pc_wo_voxel.get_center()[1] > 0))[0]
                    candidate_pc = model_pc_wo_voxel.select_by_index(candidate)
                    candidate_pc.paint_uniform_color([1, 0, 0])
                    if visualize:
                        o3d.visualization.draw_geometries([copy.copy(model_pc_wo_voxel).paint_uniform_color(self.model_color), 
                                                        candidate_pc], 
                                                        window_name='candidate_pc')
                elif class_name == 'box':
                    original_grasp_point_left = np.asarray([model_pc_wo_voxel.get_min_bound()[0], model_pc_wo_voxel.get_center()[1], model_pc_wo_voxel.get_center()[2]])
                    original_grasp_point_right = np.asarray([model_pc_wo_voxel.get_max_bound()[0], model_pc_wo_voxel.get_center()[1], model_pc_wo_voxel.get_center()[2]])
                    original_grasp_point_left_pc = o3d.geometry.PointCloud()
                    original_grasp_point_right_pc = o3d.geometry.PointCloud()
                    original_grasp_point_left_pc.points = o3d.utility.Vector3dVector([original_grasp_point_left])
                    original_grasp_point_right_pc.points = o3d.utility.Vector3dVector([original_grasp_point_right])
                    original_grasp_point_left_pc.paint_uniform_color([1, 0, 0])
                    original_grasp_point_right_pc.paint_uniform_color([1, 0, 0])

                    vis_original_grasp_point_left_pc, vis_original_grasp_point_right_pc = self.create_vis_grasp_pc(original_grasp_point_left_pc, original_grasp_point_right_pc, 10)
                    
                    if visualize:
                        o3d.visualization.draw_geometries([copy.copy(model_pc_wo_voxel).paint_uniform_color(self.model_color), 
                                                        vis_original_grasp_point_left_pc,
                                                        vis_original_grasp_point_right_pc], 
                                                        window_name='model_pc')
                        
                ################ prepare registration dataset ################
                source_pc, target_pc, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(model_pc, object_pc, voxel_size)
                if visualize:
                    o3d.visualization.draw_geometries([copy.copy(model_pc).paint_uniform_color(self.model_color), 
                                                    object_pc, 
                                                    whole_pc],
                                                    window_name='fitting model pc', 
                                                    front=self.visual_front, 
                                                    lookat=self.visual_lookat, 
                                                    up=self.visual_up,
                                                    zoom=self.visual_zoom)
                
                ################ RANSAC registration ################
                time_start = time.time()
                RANSAC_result = self.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
                RANSAC_time = time.time() - time_start
                if visualize:
                    self.draw_registration_result(source_pc, whole_pc, RANSAC_result.transformation, 'RANSAC_result')

                ################ ICP registration ################
                time_start = time.time()
                ICP_result = self.refine_registration(source_pc, target_pc, voxel_size, RANSAC_result)
                ICP_time = time.time() - time_start
                if visualize:
                    self.draw_registration_result(source_pc, whole_pc, ICP_result.transformation, 'ICP_result')

                ################ get the grasp point ################
                FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

                source_pc.transform(ICP_result.transformation)
                model_pc_wo_voxel.translate(object_center)
                model_pc_wo_voxel.transform(ICP_result.transformation)
                grasp_point_middle = model_pc_wo_voxel.get_center()

                if class_name == 'bottle':
                    candidate_pc.translate(object_center)
                    candidate_pc.transform(ICP_result.transformation)

                    model_r = Rotation.from_matrix(copy.copy(ICP_result.transformation[:3, :3]))
                    model_eular = model_r.as_euler('zxy', degrees=True)
                    print(model_eular)
                    # grasp_point_middle = candidate_pc.get_center()
                    grasp_point_left, grasp_point_right = np.zeros(3), np.zeros(3)
                    grasp_point_left = [float('inf'), float('inf'), float('inf')]
                    grasp_point_right = [-float('inf'), -float('inf'), -float('inf')]
                    for point in candidate_pc.points:
                        if model_eular[0] > -90 and model_eular[0] < 90:
                            if point[0] < grasp_point_left[0]:
                                grasp_point_left = point
                            if point[0] > grasp_point_right[0]:
                                grasp_point_right = point
                        else:
                            if point[1] < grasp_point_left[1]:
                                grasp_point_left = point
                            if point[1] > grasp_point_right[1]:
                                grasp_point_right = point
                if class_name == 'sphere':
                    # candidate_pc.translate(object_center)
                    # candidate_pc.transform(ICP_result.transformation)

                    # model_r = Rotation.from_matrix(copy.copy(ICP_result.transformation[:3, :3]))
                    # model_eular = model_r.as_euler('zxy', degrees=True)
                    # grasp_point_left, grasp_point_right = np.zeros(3), np.zeros(3)
                    # grasp_point_left = [float('inf'), float('inf'), float('inf')]
                    # grasp_point_right = [-float('inf'), -float('inf'), -float('inf')]
                    # for point in candidate_pc.points:
                    #     if model_eular[0] > -90 and model_eular[0] < 90:
                    #         if point[0] < grasp_point_left[0]:
                    #             grasp_point_left = point
                    #         if point[0] > grasp_point_right[0]:
                    #             grasp_point_right = point
                    #     else:
                    #         if point[1] < grasp_point_left[1]:
                    #             grasp_point_left = point
                    #         if point[1] > grasp_point_right[1]:
                    #             grasp_point_right = point
                    # grasp_point_left = np.array([candidate_pc.get_min_bound()[0], candidate_pc.get_center()[1], candidate_pc.get_center()[2]])
                    # grasp_point_right = np.array([candidate_pc.get_max_bound()[0], candidate_pc.get_center()[1], candidate_pc.get_center()[2]])

                    model_eular = [0.0, 0.0, 0.0]
                    grasp_point_left = np.array([model_pc_wo_voxel.get_min_bound()[0], model_pc_wo_voxel.get_center()[1], model_pc_wo_voxel.get_center()[2]])
                    grasp_point_right = np.array([model_pc_wo_voxel.get_max_bound()[0], model_pc_wo_voxel.get_center()[1], model_pc_wo_voxel.get_center()[2]])
                        
                elif class_name == 'box':
                    model_r = Rotation.from_matrix(copy.copy(ICP_result.transformation[:3, :3]))
                    model_eular = model_r.as_euler('zxy', degrees=True)
                    print(model_eular)

                    original_grasp_point_left_pc.translate(object_center)
                    original_grasp_point_right_pc.translate(object_center)
                    original_grasp_point_left_pc.transform(ICP_result.transformation)
                    original_grasp_point_right_pc.transform(ICP_result.transformation)

                    grasp_point_left = original_grasp_point_left_pc.points[0]
                    grasp_point_right = original_grasp_point_right_pc.points[0]


                grasp_point_left_pc = o3d.geometry.PointCloud()
                grasp_point_left_pc.points = o3d.utility.Vector3dVector([grasp_point_left])
                grasp_point_left_pc.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

                grasp_point_right_pc = o3d.geometry.PointCloud()
                grasp_point_right_pc.points = o3d.utility.Vector3dVector([grasp_point_right])
                grasp_point_right_pc.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

                if visualize:
                    grasp_point_middle_pc = o3d.geometry.PointCloud()
                    grasp_point_middle_pc.points = o3d.utility.Vector3dVector([grasp_point_middle])
                    grasp_point_middle_pc.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

                    vis_grasp_point_left_pc, vis_grasp_point_right_pc = self.create_vis_grasp_pc(grasp_point_left_pc, grasp_point_right_pc, 20)

                    o3d.visualization.draw_geometries([FOR1, 
                                                       source_pc, 
                                                       model_pc_wo_voxel, 
                                                       grasp_point_right_pc, 
                                                       grasp_point_left_pc, 
                                                       whole_pc], 
                                                      window_name='grasp point', 
                                                      front=self.visual_front,
                                                      lookat=self.visual_lookat,
                                                      up=self.visual_up,
                                                      zoom=self.visual_zoom)

                    FOR1 = FOR1.translate(object_center)
                    FOR1 = FOR1.scale(1, center=FOR1.get_center())
                    o3d.visualization.draw_geometries([FOR1, object_pc, 
                                                       model_pc_wo_voxel.paint_uniform_color(self.model_color),
                                                       vis_grasp_point_left_pc, 
                                                       vis_grasp_point_right_pc], 
                                                       window_name='grasp point model')
                    
                    o3d.visualization.draw_geometries([object_pc, 
                                                       model_pc_wo_voxel.paint_uniform_color(self.model_color),
                                                       vis_grasp_point_left_pc, 
                                                       vis_grasp_point_right_pc], 
                                                       window_name='grasp point model wo FOR1')
                    
                    o3d.visualization.draw_geometries([object_pc, 
                                                       model_pc_wo_voxel.paint_uniform_color(self.model_color)], 
                                                       window_name='grasp point model wo FOR1 and grasp point')
                    
                    o3d.visualization.draw_geometries([model_pc_wo_voxel, 
                                                        grasp_point_middle_pc, 
                                                        whole_pc,
                                                        vis_grasp_point_left_pc, 
                                                        vis_grasp_point_right_pc,
                                                        grasp_point_middle_pc], 
                                                        window_name='Pose estimation result',
                                                        front=self.visual_front,
                                                        lookat=self.visual_lookat,
                                                        up=self.visual_up,
                                                        zoom=self.visual_zoom)

                    # Calculate the Rotation error and Translation error
                    print('+++++++++++ Calculate the Rotation error and Translation error +++++++++++')
                    est_obj_center = np.asarray(grasp_point_middle)
                    translation_error = np.linalg.norm(est_obj_center - object_center)
                    print("Translation error(in cm): ", translation_error * 100)

                    time_start = time.time()
                    grasp_angle = self.cal_grasp_angle_from_two_grasp_points(grasp_point_left, grasp_point_right) - 90
                    grasp_angle_time = time.time() - time_start
                    print("Grasp angle: ", grasp_angle)
                    print('model_eular[0]: ', model_eular[0])
                    rotation_error = abs(grasp_angle - model_eular[0])
                    print("Rotation error(in degree): ", rotation_error)

                    print('+++++++++++ Show time consumption +++++++++++')
                    print('plane_seg_time: ', plane_seg_time)
                    print('crop_object_pc_time: ', crop_object_pc_time)
                    print('model_generate_time: ', model_generate_time)
                    print('RANSAC_time: ', RANSAC_time)
                    print('ICP_time: ', ICP_time)
                    print('grasp_angle_time: ', grasp_angle_time)

                return grasp_point_left, grasp_point_middle, grasp_point_right
            return None, None, None
                
if __name__ == '__main__':
    yolo_seg_topic = '/yolov8/instance_segment'
    result_image_topic = '/yolov8/segs_result_image'

    rs_color_topic = '/d435/color/image_raw'
    rs_depth_topic = 'd435/depth/image_raw'

    rospy.init_node('grasp', anonymous=True)
    rospy.Subscriber(yolo_seg_topic, Instance_segs, yolo_callback)
    rospy.Subscriber(result_image_topic, Image, yolo_result_callback)
    rospy.Subscriber(rs_color_topic, Image, rs_color_callback)
    rospy.Subscriber(rs_depth_topic, Image, rs_depth_callback)

    print('Start !!!')
    arm_joint = Robotic_Arm()
    gripper = moveit_commander.move_group.MoveGroupCommander('gripper')

    print(arm_joint.poseNow)
    print('current_pose', arm_joint.arm.get_current_pose())
    print('current_pose', arm_joint.poseNow)
    print('get_planning_frame', arm_joint.arm.get_planning_frame())
    arm_joint.go_named('object_placement')
    print('current_pose', arm_joint.poseNow)

    graspObject = ''
    move_obj_State = 0
    move_state = False
    home_pose = [0.1986668189072355, -0.00011792854107089447, 1.258720118031968]
    home_orientation = [0, 0, 0, 1]

    print("Running cloud...  (Listening to yolo topic:)", yolo_seg_topic)

    Pose_estimation = Pose_estimation()
    while not rospy.is_shutdown():
        if yolo_msgs is not None and yolo_result_frame is not None and yolo_msgs.instance_segs and rs_color_frame is not None and rs_depth_frame is not None and move_state == False:
            # est_success = False
            attempt_times = 0
            print('Start find grasp point----------------------------------------------------------')
            grasp_point_left, grasp_point_middle, grasp_point_right= Pose_estimation.find_grasp_point(visualize=True,
                                                                                                      rs_color_frame=rs_color_frame,
                                                                                                      rs_depth_frame=rs_depth_frame)

            print('Finish find grasp point----------------------------------------------------------')

            delta = [(0.19789000054721564 - 0.19866681890052082),
                     (0.014415581940389163 + 0.0001179397905523797),
                     (1.284 - 1.2587201180314538)]

            if grasp_point_middle is not None:
                grasp_point_middle_pixel = camera_to_pixel_coordinate(grasp_point_middle)
                grasp_point_left_pixel = camera_to_pixel_coordinate(grasp_point_left)
                grasp_point_right_pixel = camera_to_pixel_coordinate(grasp_point_right)
                
                print("grasp_point_middle_pixel: ", grasp_point_middle_pixel)
                cv2.namedWindow('rs_color_frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('rs_color_frame', 1280, 720)
                cv2.circle(rs_color_frame, (int(grasp_point_middle_pixel[0]), int(grasp_point_middle_pixel[1])), 5, (0, 0, 255), -1)
                cv2.circle(rs_color_frame, (int(grasp_point_left_pixel[0]), int(grasp_point_left_pixel[1])), 5, (0, 255, 0), -1)
                cv2.circle(rs_color_frame, (int(grasp_point_right_pixel[0]), int(grasp_point_right_pixel[1])), 5, (255, 0, 0), -1)
                cv2.line(rs_color_frame, (int(grasp_point_left_pixel[0]), int(grasp_point_left_pixel[1])), (int(
                    grasp_point_right_pixel[0]), int(grasp_point_right_pixel[1])), (0, 0, 255), 5)
                cv2.imshow('rs_color_frame', rs_color_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                [R, T] = get_RT_matrix('world', 'd435_depth_optical_frame')
                # [R_robot, T_robot] = get_RT_matrix('base_link_gr', 'd435_depth_optical_frame')
                # print(f'R: {R} \n T: {T}')

                grasp_point_left_temp = [grasp_point_left[0], grasp_point_left[1], grasp_point_left[2], 1]
                grasp_point_right_temp = [grasp_point_right[0], grasp_point_right[1], grasp_point_right[2], 1]
                grasp_point_left_w = coordinate_transform(grasp_point_left_temp, R, T)
                grasp_point_right_w = coordinate_transform(grasp_point_right_temp, R, T)
                # print("grasp_point_left_w: ", grasp_point_left_w)
                # print("grasp_point_right_w: ", grasp_point_right_w)

                ############################## cal orientation ##############################
                grasp_v = np.array(grasp_point_left_w) - np.array(grasp_point_right_w)
                target_orientation = np.array([0, 1, 0])
                norm_rt = np.linalg.norm(target_orientation)
                norm_trans = np.linalg.norm(grasp_v)
                grasp_angle = np.dot(grasp_v, target_orientation) / (norm_trans * norm_rt)
                grasp_angle = np.arccos(grasp_angle) * 180 / np.pi

                if grasp_angle > 90:
                    grasp_angle = -(180 - grasp_angle)

                if grasp_point_left[1] < grasp_point_right[1]:
                    grasp_angle = -grasp_angle

                print("grasp_angle: ", grasp_angle)
                
                grasp_axis = np.cross(grasp_v, target_orientation)
                grasp_rotation = Rotation.from_rotvec(grasp_angle * grasp_axis)
                grasp_matrix = grasp_rotation.as_matrix()
                grasp_quaternion = grasp_rotation.as_quat()

                grasp_angle_radians = np.radians(grasp_angle)
                grasp_rotation = Rotation.from_euler('z', grasp_angle_radians, degrees=False)
                current_rotation = Rotation.from_quat([home_orientation[0],
                                                    home_orientation[1],
                                                    home_orientation[2],
                                                    home_orientation[3]])
                new_grasp_rotation = grasp_rotation * current_rotation
                new_grasp_quaternion = new_grasp_rotation.as_quat()
                # print(f'new_grasp_quaternion: {new_grasp_quaternion}')

                u, v = int(grasp_point_middle_pixel[0]), int(grasp_point_middle_pixel[1])
                cameraFrame_pos = get_CameraFrame_pos(u, v, rs_depth_frame[v, u] / 1000)

                # print(f'cameraFrame_pos: {cameraFrame_pos}')
                worldFrame_pos = coordinate_transform(np.append(grasp_point_middle, 1), R, T)
                worldFrame_pos[0] -= delta[0]
                worldFrame_pos[1] -= delta[1] - 0.01
                worldFrame_pos[2] -= delta[2] - 0.2
                # print(f'worldFrame_pos: {worldFrame_pos}')

                arm_joint.go_named('home')
                waypoints = []
                waypoints.append(arm_joint.get_target_pose([worldFrame_pos[0], worldFrame_pos[1], home_pose[2]], [0, 0, 0, 1]))
                waypoints.append(arm_joint.get_target_pose([worldFrame_pos[0], worldFrame_pos[1], home_pose[2]], new_grasp_quaternion))
                waypoints.append(arm_joint.get_target_pose([worldFrame_pos[0], worldFrame_pos[1], worldFrame_pos[2]], new_grasp_quaternion))
                waypoints.append(arm_joint.get_target_pose([worldFrame_pos[0], worldFrame_pos[1], worldFrame_pos[2] - 0.02], new_grasp_quaternion))
                waypoints.append(arm_joint.get_target_pose([worldFrame_pos[0], worldFrame_pos[1], worldFrame_pos[2] - 0.04], new_grasp_quaternion))
                move_state = arm_joint.move_cartesian_by_waypoints(waypoints)

                if move_state == True:
                    gripper.set_named_target('gripper_close_half')
                    gripper.go()
                    rospy.sleep(1)

                    arm_joint.go_named('home')
                    rospy.sleep(1)
                    arm_joint.go_named('object_placement')

                    gripper.set_named_target('gripper_open')
                    gripper.go()
                    move_state = False

                rospy.sleep(1)


