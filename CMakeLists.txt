cmake_minimum_required(VERSION 3.16)
project(CudaChain CUDA)

#find_package(CUDA 9.1 REQUIRED)
#set(CMAKE_BUILD_TYPE Release)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
#set(
#        CUDA_NVCC_FLAGS
#        ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61
#)

set(CMAKE_CUDA_STANDARD 11)

set(CUDACHAIN_SOURCE_FILES
        src/CudaChain.cu
        src/CudaChain.cuh
        )

add_library(CudaChain ${CUDACHAIN_SOURCE_FILES})

#target_compile_options(CudaChain PRIVATE "SHELL:--param ggc-min-expand=1")
#target_compile_options(CudaChain PRIVATE "SHELL:--param ggc-min-heapsize=32768")

set_target_properties(
        CudaChain
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS CudaChain DESTINATION bin)
install(FILES ${CUDACHAIN_SOURCE_FILES} DESTINATION ${INCLUDE_INSTALL_DIR}/libcudachain)