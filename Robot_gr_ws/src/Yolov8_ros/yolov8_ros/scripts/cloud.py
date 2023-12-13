import torch
import rospy
import numpy as np
import cv2

from time import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header
from yolov8_ros_msgs.msg import BoundingBox, Instance_seg, Instance_segs
#color_frame = None
yolo_msgs = None



def yolo_callback(msg):
    #global color_frame
    global yolo_msgs
    yolo_msgs = msg
    # # print(color_msg.instance_segs[0].mask[0])

    # color_frame = CvBridge().imgmsg_to_cv2(msg.instance_segs[0].mask[0] ,'passthrough')
    # print(msg.instance_segs[0].boundingbox.Class)

    # return color_frame


if __name__ == '__main__':
    
    # subscribe topic
    topic_yolo ='/yolov8/instance_segment'
    # subscriber
    rospy.init_node('cloud',anonymous=True)
    rospy.Subscriber(topic_yolo, Instance_segs, yolo_callback)
    print("Running cloud...  (Listening to yolo topic:)",topic_yolo)
    
   
    
    while not rospy.is_shutdown():
        if yolo_msgs is not None:
            # print(instance_msgs.instance_segs)
            for instance_msg in yolo_msgs.instance_segs:
                if instance_msg.boundingbox.Class == 'bottle':
                    #ros use 'CvBridge().imgmsg_to_cv2' return to img
                    mask = CvBridge().imgmsg_to_cv2(instance_msg.mask[0] ,'passthrough')
                    x_min = instance_msg.boundingbox.xmin
                    y_min = instance_msg.boundingbox.ymin
                    x_max = instance_msg.boundingbox.xmax
                    y_max = instance_msg.boundingbox.ymax
                    cv2.rectangle(mask, (x_min, y_max), (x_max, y_min), (128), 2)
                    cv2.imshow('mask', mask)
                    
                    # #enter change img
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    #continue show img
                    cv2.waitKey(1)
        
            
        
    print("Ctrl-C to stop")           
            
    
    
