cmake_minimum_required(VERSION 3.8)
project(riskam_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(riskam_msgs REQUIRED)

# Install Python executables
install(PROGRAMS
  src/riskam_node.py
  src/riskam_bagger.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
