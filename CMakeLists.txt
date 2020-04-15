cmake_minimum_required(VERSION 3.5)
project(CudaChain)
find_package(CUDA REQUIRED 9.1)

if (APPLE)
    # for MacOS X
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (UNIX)
    # for Linux
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()

SET(TARGET_H CudaChain.h)

cuda_add_library(CudaChain CudaChain.cu CudaChain.h)

target_compile_options(CudaChain PRIVATE "SHELL:--param ggc-min-expand=1")
target_compile_options(CudaChain PRIVATE "SHELL:--param ggc-min-heapsize=32768")

#target_link_libraries(ConvexHull PRIVATE CudaChain)