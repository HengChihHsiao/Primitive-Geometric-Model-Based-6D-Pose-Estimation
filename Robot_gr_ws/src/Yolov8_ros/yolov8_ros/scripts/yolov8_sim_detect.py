import torch
import rospy
import numpy as np
import cv2
import time
from ultralytics import YOLO
from sensor_msgs.msg import Image
# from yolov8_ros_msgs.msg import BoundingBox,BoundingBoxes ,Masks   #new msg,name need package_msg.msg
from yolov8_ros_msgs.msg import BoundingBox, Instance_seg, Instance_segs   #new msg,name need package_msg.msg


from std_msgs.msg import Header
from cv_bridge import CvBridge


color_frame = None
store_result=[]

#############################################################################
# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class SegmentationPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        """TODO: filter by classes."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes) #class change one
        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
    

def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO object detection on an image or video source."""
    global store_result  #if global variables -> add global !
    
    model = '/home/hengchih/hengchih/Robot/Robot_gr_ws/src/Yolov8_ros/yolov8_ros/weights/3_class_sim_best.pt'
    source = color_frame
    args = dict(model=model, source=source)
    
    if use_python:
        #from ultralytics import YOLO
        # YOLO(model)(**args)
        predicter = YOLO(model=model, task='detect')
        results = predicter.predict(source=source)  #
        
        store_result=results[0]
            
    else:
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli(model=model, source=source)
        results = predictor.results

    return results[0]
        

def color_callback(color_msg):
    global color_frame
    assert isinstance(color_msg, Image)

    color_frame = CvBridge().imgmsg_to_cv2(color_msg, 'bgr8')

    return color_frame
          
    
if __name__ == '__main__':
    
    # subscribe & publisher are on while not rospy.is_shutdown():
    # subscribe topic
    topic_camera ='/d435/color/image_raw'
    # subscriber
    rospy.init_node('yolov8_ros',anonymous=True)
    rospy.Subscriber(topic_camera, Image, color_callback)
    print("Running yolov8...  (Listening to camera topic:)",topic_camera)
    
   
    #publisher topic
    topic_yolo_seg='/yolov8/instance_segment'
    topic_result_image='/yolov8/segs_result_image'
    #publisher 
    Instance_pub = rospy.Publisher(topic_yolo_seg, Instance_segs, queue_size=10)
    Seg_result_pub = rospy.Publisher(topic_result_image, Image, queue_size=10)
 
    
    while not rospy.is_shutdown():
        if color_frame is not None:
            time_start=time.time()
            results = predict(use_python=True)
            print("time:",time.time()-time_start)
            #publish
            segs_msg = Instance_segs()
            
            if results is not None:
                for result in results:
                    instance=Instance_seg()
                    
                    bbox=result.boxes.cpu().numpy()
                    # mask_img=result.masks.data.cpu().numpy().transpose(1, 2, 0)
                    instance.boundingbox.Class=result.cpu().names[bbox.cls.item()]
                    
                    xmin=int(bbox.xyxy[0][0].item())
                    ymin=int(bbox.xyxy[0][1].item())
                    xmax=int(bbox.xyxy[0][2].item())
                    ymax=int(bbox.xyxy[0][3].item())          
                    
                    instance.boundingbox.xmin=xmin
                    instance.boundingbox.ymin=ymin
                    instance.boundingbox.xmax=xmax
                    instance.boundingbox.ymax=ymax
                    instance.boundingbox.score=bbox.conf.item()
                    
                    # #sensor_msg
                    # mask=Image()
                    # mask.height = mask_img.shape[0]
                    # mask.width = mask_img.shape[1]
                    # mask.encoding = "mono8"
                    # mask.is_bigendian = False
                    # mask.step = mask.width
                    
                    # #img return rosmsg,cv2_to imgmsg can't use,because mono8 can't direct return to unit8
                    # #imgmsg is ros can read
                    # mask_img = mask_img / mask_img.max()
                    # mask_img = 255 * mask_img
                    # mask_img = mask_img.astype(np.uint8)
                    # # mask.data = mask_img.tostring()
                    # mask.data = mask_img.tobytes()
                    # instance.mask.append(mask)
                    
                    segs_msg.instance_segs.append(instance)
                
                Instance_pub.publish(segs_msg)  #at catkin_ws use " rostopic echo /yolov8/instance_segment " to look punlisher's data
                Seg_result_pub.publish(CvBridge().cv2_to_imgmsg(results.plot(), 'bgr8'))
                
    
    print("Ctrl-C to stop")
