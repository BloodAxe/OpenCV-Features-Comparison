cmake_minimum_required(VERSION 2.8)
project(OPENCV_FEATURES_COMPARISON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIR})

add_subdirectory(agast)
include_directories("${CMAKE_SOURCE_DIR}/agast")

add_subdirectory(brisk)
include_directories("${CMAKE_SOURCE_DIR}/brisk")

add_executable(test_opencv_features2d main.cpp ImageTransformation.hpp ImageTransformation.cpp FeatureAlgorithm.hpp FeatureAlgorithm.cpp AlgorithmEstimation.hpp AlgorithmEstimation.cpp CollectedStatistics.hpp
CollectedStatistics.cpp)

target_link_libraries( test_opencv_features2d agast brisk )
target_link_libraries( test_opencv_features2d ${OpenCV_LIBRARIES} )
      
install (TARGETS test_opencv_features2d DESTINATION bin)
