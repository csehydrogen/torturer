#include <cstdio>

#define CheckCUDA(exp) \
  do {\
    cudaError_t status = (exp);\
    if (status != cudaSuccess) {\
      fprintf(stderr, "[%s:%d] CUDA error: %s (code=%d)\n", \
          __FILE__, __LINE__, cudaGetErrorString(status), static_cast<int>(status));\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

const int NUM_GPU = 4;
const int BLOCK_SIZE = 64; // warp size * 2
const int BLOCK_NUM = 80 * 32; // (# of sm) * (max warp per sm / 2)
const size_t INPUT_NUM = 50790L; // (Input buffer size you want) / (BLOCK_NUM * BLOCK_SIZE * sizeof(float))

__global__ void kernel(float *in, float *out, int N, float fi0, float fi1) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float fo0 = 0, fo1 = 0;
  for (int n = 0; n < N; n += 1) {
    float fi = in[(size_t)(n % INPUT_NUM) * BLOCK_NUM * BLOCK_SIZE + gid];
    fi0 += fi;
    fi1 += fi;
    for (int c = 0; c < 11; ++c) { // memory-to-compute ratio magic number
      fo0 += fi0;
      fo1 += fi1;
    }
  }
  out[gid] = fo0 * fo1;
}

double get_time() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  long x = 1000000000L * t.tv_sec + t.tv_nsec;
  return (double)x / (double)1000000000L;
}

int main() {

  dim3 blockDim(BLOCK_SIZE, 1, 1);
  dim3 gridDim(BLOCK_NUM, 1, 1);
  #pragma omp parallel for
  for (int j = 0; j < NUM_GPU; ++j) {
    cudaSetDevice(j);
    float *in, *out;
    CheckCUDA(cudaMalloc(&in, INPUT_NUM * BLOCK_NUM * BLOCK_SIZE * sizeof(float)));
    CheckCUDA(cudaMalloc(&out, BLOCK_NUM * BLOCK_SIZE * sizeof(float)));
    while (true) {
      double st = get_time();
      kernel<<<gridDim, blockDim>>>(in, out, 1000000, 1.0, 2.0);
      cudaDeviceSynchronize();
      double et = get_time();
      printf("[Device %d] %f s\n", j, et - st);
    }
  }

  return 0;
}
