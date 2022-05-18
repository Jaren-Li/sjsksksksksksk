cmake_minimum_required(VERSION 3.16)
project(RM_Standard2022)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
# set(OpenVINO_DIR /opt/intel/openvino_2022.1.0.643/runtime/)

add_executable(RM_Standard2022
        include/rmcv/core/core.h
        include/rmcv/imgproc.h
        include/rmcv/objdetect.h
        include/rmcv/core/utils.h
        src/core.cpp
        src/imgproc.cpp
        src/utils.cpp
        main.cpp
        include/rmcv/core/serialport.h
        src/serialport.cpp
        include/rmcv/rmcv.h
        src/objdetect.cpp
        include/rmcv/core/daheng/daheng.h
        include/rmcv/core/daheng/DxImageProc.h
        include/rmcv/core/daheng/GxIAPI.h include/rmcv/mlp/mlp.h include/rmcv/mlp/training.h src/training.cpp include/rmcv/core/parallequeue.hpp include/rmcv/mlp/reasoning.h include/rmcv/debug.h src/debug.cpp)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_BINARY_DIR})

# target_link_libraries(RM_Standard2022 ${OpenCV_LIBS})
target_link_libraries(RM_Standard2022
                      PRIVATE
                      openvino::runtime 
                      gxiapi 
                      pthread
                      ${OpenCV_LIBS}
                      )
