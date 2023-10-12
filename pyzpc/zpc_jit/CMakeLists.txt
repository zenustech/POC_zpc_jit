cmake_minimum_required(VERSION 3.18)

project(pyzpc LANGUAGES C CXX)
find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)
set(ZS_BUILD_SHARED_LIBS OFF)
set(WHEREAMI_BUILD_SHARED_LIBS ON)
add_subdirectory(zpc)
