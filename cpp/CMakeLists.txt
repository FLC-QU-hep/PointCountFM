cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(FastShowerFlow)

find_package(Torch REQUIRED)
find_package(HDF5 COMPONENTS CXX REQUIRED)

set(C_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(inference inference.cc)
target_link_libraries(inference PRIVATE ${HDF5_CXX_LIBRARIES})
target_include_directories(inference PRIVATE ${HDF5_CXX_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE "${TORCH_LIBRARIES}")
