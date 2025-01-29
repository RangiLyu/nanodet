cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)
project(onnxruntime_demo)

find_package(OpenCV REQUIRED)

find_package(onnxruntime REQUIRED)
include_directories(${onnxruntime_INCLUDE_DIRS}/onnxruntime/include)
if(NOT TARGET onnxruntime)
    message(WARNING "onnxruntime NOT FOUND!  Please set onnxruntime_DIR environment variable")
else()
    message("onnxruntime FOUND ")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(onnxruntime_demo main.cpp nanodet.cpp)

target_link_libraries(
    onnxruntime_demo
    onnxruntime
    ${OpenCV_LIBS}
    ${onnxruntime_LIBRARY}
)
