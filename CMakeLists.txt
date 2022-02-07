cmake_minimum_required(VERSION 3.10)

project(ipp_panoptic_mapping)

# Resolve dependencies for this subproject
find_package(PCL 1.10 REQUIRED COMPONENTS common io features visualization REQUIRED)
find_package(PkgConfig REQUIRED) # Required to link againts jsoncpp
pkg_check_modules(JSONCPP jsoncpp)

### cloud_to_segs ###
set(CLOUD_TO_SEGS "cloud_to_segs")
add_executable(${CLOUD_TO_SEGS} ${CMAKE_CURRENT_SOURCE_DIR}/cloud_to_segs/cloud_to_segs.cpp)
target_compile_features(${CLOUD_TO_SEGS} PRIVATE cxx_std_17)
target_compile_definitions(${CLOUD_TO_SEGS} PRIVATE ${PCL_DEFINITIONS})
target_include_directories(${CLOUD_TO_SEGS} PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(${CLOUD_TO_SEGS} ${PCL_LIBRARIES} ${JSONCPP_LIBRARIES})