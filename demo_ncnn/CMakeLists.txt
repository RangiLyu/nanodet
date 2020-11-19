cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)

project(nanodet_demo)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

find_package(Vulkan REQUIRED)
message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

set(NCNN_LIBS D:/Programs/ncnn/build-vs2019/install/lib)
set(NCNN_INCLUDE_DIRS D:/Programs/ncnn/build-vs2019/install/include)


include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

link_directories(
    ${NCNN_LIBS}
)

add_executable(nanodet_demo main.cpp nanodet.cpp)

target_link_libraries(
    nanodet_demo 
    ncnn.lib
    glslang.lib
    OGLCompiler.lib
    OSDependent.lib
    SPIRV.lib
    ${Vulkan_LIBRARY}
    ${OpenCV_LIBS} 
)