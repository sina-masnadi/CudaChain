cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER "/usr/bin/gcc-6")
set(CMAKE_CXX_COMPILER "/usr/bin/gcc-6")
project(CudaChain)
find_package(CUDA REQUIRED 9.1)
set(CMAKE_BUILD_TYPE Release)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61
)

include_directories(src)

set(CUDACHAIN_SOURCE_FILES
        src/CudaWrapper.h
        src/ThrustWrapper.h
        src/CudaChain.h
        src/CudaChain.cu
        )

CUDA_ADD_LIBRARY(
        ${PROJECT_NAME}
        ${CUDACHAIN_SOURCE_FILES}
)

target_compile_options(CudaChain PRIVATE "SHELL:--param ggc-min-expand=1")
target_compile_options(CudaChain PRIVATE "SHELL:--param ggc-min-heapsize=32768")
#
install(TARGETS CudaChain DESTINATION bin)
install(FILES ${CUDACHAIN_SOURCE_FILES} DESTINATION ${INCLUDE_INSTALL_DIR}/libcudachain)
