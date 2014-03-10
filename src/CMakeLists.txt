cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(voxelQuality)

set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions("-Dqh_QHpointer")

FIND_PACKAGE(OpenCV REQUIRED)

file(MAKE_DIRECTORY ${../bin/})

add_executable(../bin/voxelQuality main.cpp voxelizer.cpp camera.cpp RayCaster.cpp IO.cpp utils.cpp)
target_link_libraries (../bin/voxelQuality ${PCL_LIBRARIES} ${OpenCV_LIBS})
