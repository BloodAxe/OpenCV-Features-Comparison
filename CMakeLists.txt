cmake_minimum_required(VERSION 2.8)
project(OPENCV_FEATURES_COMPARISON)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

include_directories(${OpenCV_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIR})

add_executable(test_opencv_features2d main.cpp ImageTransformation.hpp ImageTransformation.cpp FeatureAlgorithm.hpp FeatureAlgorithm.cpp AlgorithmEstimation.hpp AlgorithmEstimation.cpp CollectedStatistics.hpp
CollectedStatistics.cpp)

target_link_libraries( test_opencv_features2d ${OpenCV_LIBRARIES} )
      
install (TARGETS test_opencv_features2d DESTINATION bin)
