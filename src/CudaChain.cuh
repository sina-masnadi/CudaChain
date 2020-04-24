#include "CudaWrapper.h"
#include "ThrustWrapper.h"


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