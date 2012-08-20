add_library(brisk hammingsse.hpp brisk.h brisk.cpp hammingsse.hpp)

target_link_libraries(brisk agast)
include_directories("${CMAKE_SOURCE_DIR}")
