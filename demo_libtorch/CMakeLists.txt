cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)

project(nanodet_demo)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(nanodet_demo main.cpp nanodet_libtorch.cpp)

target_link_libraries(
    nanodet_demo
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET nanodet_demo
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:nanodet_demo>)
endif (MSVC)
