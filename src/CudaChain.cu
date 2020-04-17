// Implement CudaChain convex hull algorithm on the GPU
// Original:
// Gang Mei, Institute of Earth and Enviromental Sciences
//           University of Freiburg, April 8, 2014
//           gang.mei@geologie.uni-freiburg.de
//           gangmeiphd@gmail.com
// Revised:
// Gang Mei, School of Engineeing and Technology,
//           China University of Geosciences, December 27, 2015
//           gang.mei@cugb.edu.cn
//           gangmeiphd@gmail.com
// Warning:  Ignore numerical non-robustness and geometrical degeneracies


#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <ctime>

#include "CudaChain.cuh"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/////////////// Input & Output Interface ////////////////

// Input: Use the QHull input forrmat, i.e.,
/* Dimension  Number of points
// Corrdinates of Point 1
// Coordinates of Point 2
   ...                   */
void importQHull(thrust::host_vector<float> &pts_x,
                 thrust::host_vector<float> &pts_y,
                 std::string &path) {
    char str1[20];
//    cout <<"\nPlease Input File Name of Point Set:  ";
//    cin >>str1;
    ifstream fin(path.c_str());
    if (!fin) {
        cout << "\nCannot Open File ! " << str1 << endl;
        exit(1);
    }

    float x, y;
    int Dim, nVertex;
    fin >> Dim >> nVertex;
    pts_x.resize(nVertex);
    pts_y.resize(nVertex);

    for (int i = 0; i < nVertex; i++) {
        fin >> x >> y;
        pts_x[i] = x;
        pts_y[i] = y;
    }
    fin.close();

}

// Export the resulting convex hull to OFF file
void exportOFF(CudaChainPoint *pts, int n) // Use C/C++ arrays for exporting
{
    char str2[20];
    cout << "\nPlease Input File Name of Convex Hull:  ";
    cin >> str2;
    ofstream fout(str2);
    if (!fout) {
        cout << "\nCannot Save File ! " << str2 << endl;
        exit(1);
    }

    float x, y, z;
    int nVert = n;

    fout << "OFF" << endl;
    fout << nVert << "  " << 1 << "  " << 0 << endl;
    for (int i = 0; i < nVert; i++) {
        x = pts[i].x;
        y = pts[i].y;
        z = 0.0;
        fout << x << "   " << y << "   " << z << endl;
    }
    fout << nVert << "   ";

    for (int i = 0; i < nVert; i++) {
        fout << i << "   ";
    }

    fout.close();

}

// Export the resulting convex hull to OFF file
void exportOFF(CudaChainPoint **pts, int n) // Use C/C++ arrays for exporting
{
    char str2[20];
    cout << "\nPlease Input File Name of Convex Hull:  ";
    cin >> str2;
    ofstream fout(str2);
    if (!fout) {
        cout << "\nCannot Save File ! " << str2 << endl;
        exit(1);
    }

    double x, y, z;
    int ids[3];

    int nVert = n;

    fout << "OFF" << endl;
    fout << nVert << "  " << 1 << "  " << 0 << endl;
    for (int i = 0; i < nVert; i++) {
        x = pts[i]->x;
        y = pts[i]->y;
        z = 0.0;
        fout << x << "   " << y << "   " << z << endl;
    }
    fout << nVert << "   ";

    for (int i = 0; i < nVert; i++) {
        fout << i << "   ";
    }

    fout.close();

}

// Export the resulting convex hull to OFF file
void exportOFF(const thrust::host_vector<float> &pts_x,
               const thrust::host_vector<float> &pts_y, int n) // Use thrust containers for exporting
{
    char str2[20];
    cout << "\nPlease Input File Name of Convex Hull:  ";
    cin >> str2;
    ofstream fout(str2);
    if (!fout) {
        cout << "\nCannot Save File ! " << str2 << endl;
        exit(1);
    }

    float x, y, z;
    int nVert = n;

    fout << "OFF" << endl;
    fout << nVert << "  " << 1 << "  " << 0 << endl;
    for (int i = 0; i < nVert; i++) {
        x = pts_x[i];
        y = pts_y[i];
        z = 0.0;
        fout << x << "   " << y << "   " << z << endl;
    }
    fout << nVert << "   ";

    for (int i = 0; i < nVert; i++) {
        fout << i << "   ";
    }

    fout.close();

}

/////////////////////////////////////////////////////////
// isLeft():tests if a point is Left|On|Right of an infinite line.
// Input: three points P0, P1, and P2
// Return:
//	>0 for P2 left of the line through P0 and P1
// =0 for P2 on the line
// <0 for P2 right of the line
// See: Algorithm 1 on Area of Triangles
__host__ __device__

inline float isLeft(CudaChainPoint *P0, CudaChainPoint *P1, CudaChainPoint *P2) {
    return (P1->x - P0->x) * (P2->y - P0->y) - (P2->x - P0->x) * (P1->y - P0->y);
}

__host__ __device__

inline float isLeft(CudaChainPoint *P0, CudaChainPoint *P1, float &P2_x, float &P2_y) {
    return (P1->x - P0->x) * (P2_y - P0->y) - (P2_x - P0->x) * (P1->y - P0->y);
}

// Check whether a point locates in a convex quadrilateral, for 1st discarding
__host__ __device__

int isInside(CudaChainPoint *P0, CudaChainPoint *P1, CudaChainPoint *P2, CudaChainPoint *P3, CudaChainPoint *pt) {
    if (isLeft(P0, P1, pt) <= 0.0) return 1; // In the sub-region 1
    if (isLeft(P1, P2, pt) <= 0.0) return 2; // In the sub-region 2
    if (isLeft(P2, P3, pt) <= 0.0) return 3; // In the sub-region 3
    if (isLeft(P3, P0, pt) <= 0.0) return 4; // In the sub-region 4

    return 0;                                // In the sub-region 0
}

__host__ __device__

int isInside(CudaChainPoint *P0, CudaChainPoint *P1, CudaChainPoint *P2, CudaChainPoint *P3, float &pt_x, float &pt_y) {
    if (isLeft(P0, P1, pt_x, pt_y) <= 0.0) return 1;
    if (isLeft(P1, P2, pt_x, pt_y) <= 0.0) return 2;
    if (isLeft(P2, P3, pt_x, pt_y) <= 0.0) return 3;
    if (isLeft(P3, P0, pt_x, pt_y) <= 0.0) return 4;

    return 0;
}

// Preprocess by discarding interior points, i.e., 1st round of discarding
__global__
void kernelPreprocess(float *h_extreme_x, float *h_extreme_y,
                      float *v_x, float *v_y, int *pos, int n) {
    __shared__ float2 s_extreme[4]; // Stored in shared memory

    if (threadIdx.x == 0) {
        for (int t = 0; t < 4; t++) {
            s_extreme[t].x = h_extreme_x[t];
            s_extreme[t].y = h_extreme_y[t];
        }
    }
    __syncthreads();

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) // Check
    {
        pos[i] = isInside(&s_extreme[0], &s_extreme[1],
                          &s_extreme[2], &s_extreme[3], v_x[i], v_y[i]);
    }
}


// Make predicate easy to write
typedef thrust::tuple<float, float, int> FloatTuple3;

// Predicate
struct is_interior_tuple {
    __host__ __device__

    bool operator()(const FloatTuple3 &p) {
        return thrust::get<2>(p) > 0;
    }
};

struct is_region_1_tuple {
    __host__ __device__

    bool operator()(const FloatTuple3 &p) {
        return thrust::get<2>(p) == 1;
    }
};

struct is_region_2_tuple {
    __host__ __device__

    bool operator()(const FloatTuple3 &p) {
        return thrust::get<2>(p) == 2;
    }
};

struct is_region_3_tuple {
    __host__ __device__

    bool operator()(const FloatTuple3 &p) {
        return thrust::get<2>(p) == 3;
    }
};

struct is_region_4_tuple {
    __host__ __device__

    bool operator()(const FloatTuple3 &p) {
        return thrust::get<2>(p) == 4;
    }
};


// Discard invalid points in Region 1 (i.e., the lower left)
__global__
void kernelCheck_R1_device(float *V, int *pos, int base, int length) {
    int i, id;
    float temp;
    int items = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (BLOCK_SIZE == 1024) {
        temp = V[base];
        for (int t = 0; t < items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] > temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 512) {
        temp = V[base];
        for (int t = 0; t < 2 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] > temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 256) {
        temp = V[base];
        for (int t = 0; t < 4 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] > temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

}

// Discard invalid points in Region 2 (i.e., the lower right)
__global__
void kernelCheck_R2_device(float *V, int *pos, int base, int length) {
    int i, id;
    float temp;
    int items = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (BLOCK_SIZE == 1024) {
        temp = V[base];
        for (int t = 0; t < items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] < temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 512) {
        temp = V[base];
        for (int t = 0; t < 2 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] < temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 256) {
        temp = V[base];
        for (int t = 0; t < 4 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] < temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

}

// Discard invalid points in Region 3 (i.e., the upper right)
__global__
void kernelCheck_R3_device(float *V, int *pos, int base, int length) {
    int i, id;
    float temp;
    int items = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (BLOCK_SIZE == 1024) {
        temp = V[base];
        for (int t = 0; t < items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] < temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 512) {
        temp = V[base];
        for (int t = 0; t < 2 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] < temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 256) {
        temp = V[base];
        for (int t = 0; t < 4 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] < temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

}

// Discard invalid points in Region 4 (i.e., the upper left)
__global__
void kernelCheck_R4_device(float *V, int *pos, int base, int length) {
    int i, id;
    float temp = V[base];
    int items = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (BLOCK_SIZE == 1024) {
        temp = V[base];
        for (int t = 0; t < items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] > temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 512) {
        temp = V[base];
        for (int t = 0; t < 2 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] > temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

    if (BLOCK_SIZE == 256) {
        temp = V[base];
        for (int t = 0; t < 4 * items; t++) {
            i = threadIdx.x * items + t;
            id = i + base;
            if (i < length && pos[id] > 0) {
                if (V[id] > temp) pos[id] = 0;
                else temp = V[id];
            }
        }
    }

}


// Wrapper: CudaChain
int cudaChainNew(const thrust::host_vector<float> &x, const thrust::host_vector<float> &y,
                 thrust::host_vector<float> &hull_x, thrust::host_vector<float> &hull_y) {
    int n = x.size(); // Number of points

    // Copy data from host side to device side
    thrust::device_vector<float> d_x = x;
    thrust::device_vector<float> d_y = y;

    typedef thrust::device_vector<float>::iterator FloatIter;

    // Find the four extreme points, i.e., min x, max x, min y, and max y
    thrust::pair<FloatIter, FloatIter> extremex = thrust::minmax_element(d_x.begin(), d_x.end());
    thrust::pair<FloatIter, FloatIter> extremey = thrust::minmax_element(d_y.begin(), d_y.end());

    // One method to find min / max (Correct)
    thrust::device_vector<float>::iterator minx = extremex.first;
    thrust::device_vector<float>::iterator maxx = extremex.second;
    thrust::device_vector<float>::iterator miny = extremey.first;
    thrust::device_vector<float>::iterator maxy = extremey.second;

    /* Another method to find min / max (Correct too)
    thrust::device_vector<float>::iterator minx = thrust::min_element(d_x.begin(), d_x.end());
    thrust::device_vector<float>::iterator maxx = thrust::max_element(d_x.begin(), d_x.end());
    thrust::device_vector<float>::iterator miny = thrust::min_element(d_y.begin(), d_y.end());
    thrust::device_vector<float>::iterator maxy = thrust::max_element(d_y.begin(), d_y.end());
    */

    // Store the four extreme points temporarily
    thrust::device_vector<float> d_extreme_x(4);
    thrust::device_vector<float> d_extreme_y(4);

    d_extreme_x[0] = *minx;
    d_extreme_x[1] = d_x[miny - d_y.begin()];
    d_extreme_x[2] = *maxx;
    d_extreme_x[3] = d_x[maxy - d_y.begin()];

    d_extreme_y[0] = d_y[minx - d_x.begin()];
    d_extreme_y[1] = *miny;
    d_extreme_y[2] = d_y[maxx - d_x.begin()];
    d_extreme_y[3] = *maxy;

    thrust::device_vector<int> d_pos(n); // Indicators for points' position

    // Get the pointers to the arrays, to be used as launch arguments
    float *d_extreme_x_ptr = thrust::raw_pointer_cast(&d_extreme_x[0]);
    float *d_extreme_y_ptr = thrust::raw_pointer_cast(&d_extreme_y[0]);
    float *d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
    float *d_y_ptr = thrust::raw_pointer_cast(&d_y[0]);
    int *d_pos_ptr = thrust::raw_pointer_cast(&d_pos[0]);

    // 1st discarding :  Block size can be 1024 in this kernel
    kernelPreprocess << < (n + 1023) / 1024, 1024 >> > (d_extreme_x_ptr, d_extreme_y_ptr,
            d_x_ptr, d_y_ptr, d_pos_ptr, n);

    // Set Extreme Points Specifically
    d_pos[minx - d_x.begin()] = 1;  // min X
    d_pos[miny - d_y.begin()] = 2;  // min Y
    d_pos[maxx - d_x.begin()] = 3;  // max X
    d_pos[maxy - d_y.begin()] = 4;  // max Y

    // Defining a zip_iterator type can be a little cumbersome ...
    typedef thrust::device_vector<float>::iterator FloatIterator;
    typedef thrust::device_vector<int>::iterator IntIterator;
    typedef thrust::tuple<FloatIterator, FloatIterator, IntIterator> FloatIteratorTuple;
    typedef thrust::zip_iterator<FloatIteratorTuple> Float3Iterator;

    // create some zip_iterators
    Float3Iterator P_first = thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin(), d_pos.begin()));
    Float3Iterator P_last = thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end(), d_pos.end()));

    // pass the zip_iterators into Partion()
    // Partion
    Float3Iterator first_of_R0 = thrust::partition(P_first, P_last,
                                                   is_interior_tuple());                   // Find Interior
    Float3Iterator first_of_R2 = thrust::partition(P_first, first_of_R0 - 1,
                                                   is_region_1_tuple());            // Find Region 1
    Float3Iterator first_of_R3 = thrust::partition(first_of_R2, first_of_R0 - 1,
                                                   is_region_2_tuple());        // Find Region 2
    Float3Iterator first_of_R4 = thrust::partition(first_of_R3, first_of_R0 - 1,
                                                   is_region_3_tuple());        // Find Region 3
//	CudaChainPoint * first_of_R4 = thrust::partition(first_of_R3, first_of_R0-1, is_region_4());   Not needed        // Find Region 4

    Float3Iterator first_of_R1 = P_first;

    // Get the position of the first points in each sub-region
    FloatIteratorTuple pos_R1 = first_of_R1.get_iterator_tuple();
    FloatIteratorTuple pos_R2 = first_of_R2.get_iterator_tuple();
    FloatIteratorTuple pos_R3 = first_of_R3.get_iterator_tuple();
    FloatIteratorTuple pos_R4 = first_of_R4.get_iterator_tuple();
    FloatIteratorTuple pos_R0 = first_of_R0.get_iterator_tuple();

    // Partly Sort for each sub-regions
    // Region 1 : ascending X
    FloatIterator first_of_R1_x = thrust::get<0>(pos_R1);
    FloatIterator first_of_R2_x = thrust::get<0>(pos_R2);
    thrust::sort_by_key(first_of_R1_x, first_of_R2_x, first_of_R1);

    // Region 2 : ascending Y
    FloatIterator first_of_R2_y = thrust::get<1>(pos_R2);
    FloatIterator first_of_R3_y = thrust::get<1>(pos_R3);
    thrust::sort_by_key(first_of_R2_y, first_of_R3_y, first_of_R2);

    // Region 3 : descending X
    FloatIterator first_of_R3_x = thrust::get<0>(pos_R3);
    FloatIterator first_of_R4_x = thrust::get<0>(pos_R4);
    thrust::sort_by_key(first_of_R3_x, first_of_R4_x, first_of_R3);
    // Sort in ascending order, and then reverse
    thrust::reverse(thrust::get<0>(pos_R3), thrust::get<0>(pos_R4));
    thrust::reverse(thrust::get<1>(pos_R3), thrust::get<1>(pos_R4));
    thrust::reverse(thrust::get<2>(pos_R3), thrust::get<2>(pos_R4));

    // Region 4 : descending Y
    FloatIterator first_of_R4_y = thrust::get<1>(pos_R4);
    FloatIterator first_of_R0_y = thrust::get<1>(pos_R0);
    thrust::sort_by_key(first_of_R4_y, first_of_R0_y, first_of_R4);
    // Sort in ascending order, and then reverse
    thrust::reverse(thrust::get<0>(pos_R4), thrust::get<0>(pos_R0));
    thrust::reverse(thrust::get<1>(pos_R4), thrust::get<1>(pos_R0));
    thrust::reverse(thrust::get<2>(pos_R4), thrust::get<2>(pos_R0));

    // Kernel : 2nd round of discarding
    d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
    d_y_ptr = thrust::raw_pointer_cast(&d_y[0]);
    d_pos_ptr = thrust::raw_pointer_cast(&d_pos[0]);

    int base, length;

    // For each sub-region:  Single Block and Single Pass
    // Region 1
    base = 0;
    length = first_of_R2 - first_of_R1;
    kernelCheck_R1_device << < 1, BLOCK_SIZE >> > (d_y_ptr, d_pos_ptr, base, length);  // Only Y

    // Region 2
    base = first_of_R2 - first_of_R1;
    length = first_of_R3 - first_of_R2;
    kernelCheck_R2_device << < 1, BLOCK_SIZE >> > (d_x_ptr, d_pos_ptr, base, length);  // Only X

    // Region 3
    base = first_of_R3 - first_of_R1;
    length = first_of_R4 - first_of_R3;
    kernelCheck_R3_device << < 1, BLOCK_SIZE >> > (d_y_ptr, d_pos_ptr, base, length);  // Only Y

    // Region 4
    base = first_of_R4 - first_of_R1;
    length = first_of_R0 - first_of_R4;
    kernelCheck_R4_device << < 1, BLOCK_SIZE >> > (d_x_ptr, d_pos_ptr, base, length);  // Only X

    // Re-Partion, Note that: Use stable partition, rather than partition
    Float3Iterator first_of_invalid = thrust::stable_partition(first_of_R1, first_of_R0, is_interior_tuple());

    // Copy
    n = first_of_invalid - first_of_R1;
    thrust::copy_n(thrust::get<0>(pos_R1), n, hull_x.begin());
    thrust::copy_n(thrust::get<1>(pos_R1), n, hull_y.begin());

    return n;

}


// simpleHull_2D(): Melkman convex hull algorithm for simple polygon
//    Input:  V[] = polyline array of 2D vertex points (Note : CCW)
//            n   = the number of points in V[]
//    Output: H[] = output convex hull array of vertices (max is n)
//    Return: h   = the number of points in H[]
__host__ __device__

int simpleHull_2D(CudaChainPoint *V, int n, CudaChainPoint *H) {
    // initialize a deque D[] from bottom to top so that the
    // 1st three vertices of V[] are a counterclockwise triangle
    CudaChainPoint *D = new CudaChainPoint[2 * n + 1];
    int bot = n - 2, top = bot + 3;   // initial bottom and top deque indices
    D[bot] = D[top] = V[2];       // 3rd vertex is at both bot and top
    if (isLeft(&V[0], &V[1], &V[2]) > 0) {
        D[bot + 1] = V[0];
        D[bot + 2] = V[1];          // ccw vertices are: 2,0,1,2
    } else {
        D[bot + 1] = V[1];
        D[bot + 2] = V[0];          // ccw vertices are: 2,1,0,2
    }

    // compute the hull on the deque D[]
    for (int i = 3; i < n; i++)     // process the rest of vertices
    {
        // test if next vertex is inside the deque hull
        if ((isLeft(&D[bot], &D[bot + 1], &V[i]) > 0) &&
            (isLeft(&D[top - 1], &D[top], &V[i]) > 0))
            continue;         // skip an interior vertex

        // incrementally add an exterior vertex to the deque hull
        // get the rightmost tangent at the deque bot
        while (isLeft(&D[bot], &D[bot + 1], &V[i]) <= 0)
            ++bot;                // remove bot of deque
        D[--bot] = V[i];          // insert V[i] at bot of deque

        // get the leftmost tangent at the deque top
        while (isLeft(&D[top - 1], &D[top], &V[i]) <= 0)
            --top;                // pop top of deque
        D[++top] = V[i];          // push V[i] onto top of deque
    }

    // transcribe deque D[] to the output hull array H[]
    int h;        // hull vertex counter
    for (h = 0; h <= (top - bot); h++)
        H[h] = D[bot + h];

    delete D;
    return h - 1;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    std::cout << "\nBLOCK_SIZE :  " << BLOCK_SIZE << '\n';

    // Host containers for storing x and y
    thrust::host_vector<float> ipts_x;
    thrust::host_vector<float> ipts_y;


    std::string path = "/home/sina/Documents/ConvexHullPlayGround/cudachain/sample.txt";
    // Input points
    importQHull(ipts_x, ipts_y, path);
    cout << ipts_x.size() << endl;
    for (float pt:ipts_x) {
        cout << pt << endl;
    }

    int n = ipts_x.size();

    thrust::host_vector<float> opts_x(n);
    thrust::host_vector<float> opts_y(n);

    // Perform 1st and 2nd rounds of discarding
    int h = cudaChainNew(ipts_x, ipts_y, opts_x, opts_y);

    // Finalize the computing of expected convex hull, i.e.,
    // Calculate the convex hull of a simple polygon
    CudaChainPoint *opts = new CudaChainPoint[h];
    for (int i = 0; i < h; i++) {
        opts[i].x = opts_x[i];
        opts[i].y = opts_y[i];
    }
    CudaChainPoint *H = new CudaChainPoint[h];
    h = simpleHull_2D(opts, h, H);

    // Output: to OFF file format
//    exportOFF(H, h);

    delete[] opts;

//    system("pause");

    return 0;
}
