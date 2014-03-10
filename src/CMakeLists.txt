cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(voxelProcessing)

set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions("-Dqh_QHpointer")

FIND_PACKAGE(OpenCV REQUIRED)

file(MAKE_DIRECTORY ${../bin/})

add_executable(../bin/voxelProcessing main.cpp voxelizer.cpp camera.cpp RayCaster.cpp IO.cpp utils.cpp)
target_link_libraries (../bin/voxelProcessing ${PCL_LIBRARIES} ${OpenCV_LIBS})
