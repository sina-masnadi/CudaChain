#ifndef CUDACHAIN_LIBRARY_CUH
#pragma once

#include <device_launch_parameters.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>

#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/reverse.h>
#include <thrust/pair.h>

#define CUDACHAIN_LIBRARY_CUH

using namespace std;

#define BLOCK_SIZE 1024
#define CudaChainPoint  float2

namespace CudaChain {
    void importQHull(thrust::host_vector<float> &pts_x,
                     thrust::host_vector<float> &pts_y,
                     std::string &path);

    int cudaChainNew(const thrust::host_vector<float> &x,
                     const thrust::host_vector<float> &y,
                     thrust::host_vector<float> &hull_x,
                     thrust::host_vector<float> &hull_y);

    __host__ __device__
    int simpleHull_2D(CudaChainPoint *V, int n, CudaChainPoint *H);

    void exportOFF(CudaChainPoint *pts, int n);
}

#endif //CUDACHAIN_LIBRARY_CUH
