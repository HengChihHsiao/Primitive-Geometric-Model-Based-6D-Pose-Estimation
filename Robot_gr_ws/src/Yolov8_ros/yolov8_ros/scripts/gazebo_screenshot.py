import rospy
import numpy as np
import cv2
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


color_frame = None

def color_callback(color_msg):
    global color_frame
    assert isinstance(color_msg, Image)

    color_frame = CvBridge().imgmsg_to_cv2(color_msg, 'bgr8')

    return color_frame

if __name__ == '__main__':
    camera_topic = '/d435/color/image_raw'
    screenshot_folder = '/home/hengchih/hengchih/Robot/Robot_gr_ws/src/Yolov8_ros/yolov8_ros/gazebo_screenshot/ball_mix'

    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    rospy.init_node('yolov8_ros', anonymous=True)
    rospy.Subscriber(camera_topic, Image, color_callback)

    screenshot_count = 0
    while not rospy.is_shutdown():
        if color_frame is not None:
            cv2.namedWindow('color', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('color', 1280, 720)
            cv2.imshow('color', color_frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(os.path.join(screenshot_folder, f'{screenshot_folder.split("/")[-1]}_{screenshot_count}.png'), color_frame)
                screenshot_count += 1
                print('Screenshot saved.')
            elif key == ord('q'):
                break
            # elif key == ord('c'):
            #     continue
