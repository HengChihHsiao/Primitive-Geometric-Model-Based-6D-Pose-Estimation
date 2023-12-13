cmake_minimum_required(VERSION 3.0.2)
project(yolov8_ros_msgs)


find_package(catkin REQUIRED COMPONENTS 
  sensor_msgs
  std_msgs
  message_generation
  
 
)

add_message_files(
        DIRECTORY msg
        FILES
        BoundingBox.msg
        Instance_seg.msg
        Instance_segs.msg
        
)

generate_messages(
        DEPENDENCIES std_msgs sensor_msgs
        
)

catkin_package(

)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
