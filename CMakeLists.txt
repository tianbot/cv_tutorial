cmake_minimum_required(VERSION 2.8.3)
project(cv_tutorial)
#set(OpenCV_DIR /usr/local/share/OpenCV)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
)
find_package(OpenCV REQUIRED)
catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(main_test   src/main_test.cpp)
target_link_libraries(main_test ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
