**Robotic Grasping Using Object Detection and Primitive Geometric Model Based 6D Pose Estimation**
=======

<p align="center">
	<img src ="6D_pose_est_arch.png" width="800" />
</p>

## Table of Content
- [Overview](#overview)
- [Requirements](#requirements)
- [Code Structure](#code-structure)
- [Setup](#setup)
- [Results](#results)
- [Citations](#citations)
- [License](#license)

## Overview

This repository is the official implementation of [Robotic Grasping Using Object Detection and Primitive Geometric Model Based 6D Pose Estimation]([Video]()) at [Robot Vision Lab](https://vision.ee.ccu.edu.tw/).

Our method is a two-stage pipeline that first detects the object and then estimates the 6D pose of the object using a primitive geometric model.

The ROS code of the simulation environment using Gazebo is inculded in this repository.

## Requirements

* ROS Noetic
* Ubuntu 20.04
* Python 3.8
* Pytorch 1.13.1
* open3d
* ultralytics
* moveit_commander
* opencv-python
* rospy
* ros_numpy
* numpy
* cv_bridge
* scipy
* tf

## Code Structure
<!-- 
```
├── README.md
├── 6D_pose_est_arch.png
├── Robot_gr_ws
│   ├── src
│   │   ├── handover_robot
│   │   │   ├── robot_description (URDF files of the robot)
│   │   │   ├── robot_gazebo (Including the main pose estimation code & launch files)
│   │   │   ├── robot_gripper_moveit_config (Moveit config files)
``` -->

* **Robot_gr_ws/src**
    * **handover_robot**
        * **robot_description**: URDF files of the robot
        * **robot_gazebo**: Including the main pose estimation code & launch files
            * **scripts/Pose_est.py**: Main pose estimation code
            * **launch/robot_gr_bringup_moveit.launch**: The main launch file
        * **robot_gripper_moveit_config**: Moveit config files
    * **Yolov8_ros**: Object detection ROS package
        * **yolov8_ros** 
            * **scripts/yolov8_sim_detect.py**: Object detection code for simulation, you can modify the rgb adn depth topic in this file
            * **weights**: The weights of the Yolov8 model


## Setup

1. Clone this repository
   ```
   git clone https://github.com/HengChihHsiao/Primitive-Geometric-Model-Based-6D-Pose-Estimation.git
   ```
2. Install ROS Noetic
   ```
    http://wiki.ros.org/noetic/Installation/Ubuntu
    ```
3. Install the required packages
    ```
    pip install -r requirements.txt
    ```
4. Install the required ROS packages
    ```
    sudo apt-get install ros-noetic-moveit ros-noetic-ros-control ros-noetic-ros-controllers
    sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control ros-noetic-ros-numpy
    ```
5. build the ROS workspace

    cd into the workspace
    ```
    cd Robot_gr_ws
    ```
    build the workspace, initialize the workspace
    ```
    catkin_init_workspace
    ```
    build the workspace
    ```
    catkin_build
    ```
    source the workspace
    ```
    source devel/setup.bash
    ```
6. launch the simulation environment
    ```
    roslaunch handover_robot robot_gr_bringup_moveit.launch
    ```

7. launch yolov8 object detection
    
    You can download our trained weights for some Gazebo objects[here]()
    ```
    roslaunch yolov8_ros yolov8_ros.launch
    ```
    You can modify the rgb topic in the yolov8_ros/scripts/yolov8_sim_detect.py file

8. execute the pose estimation code

    ```
    roscd handover_robot/robot_gazebo/scripts
    python3 Pose_est.py
    ```
    
## Results
