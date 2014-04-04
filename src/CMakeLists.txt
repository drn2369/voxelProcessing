
FIND_PACKAGE(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions("-Dqh_QHpointer")

FIND_PACKAGE(OpenCV REQUIRED)

set(dir ${CMAKE_CURRENT_DIR}../bin/)
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

add_executable(voxelProcessing main.cpp voxelizer.cpp camera.cpp RayCaster.cpp IO.cpp utils.cpp)
target_link_libraries (voxelProcessing ${PCL_LIBRARIES} ${OpenCV_LIBS})
