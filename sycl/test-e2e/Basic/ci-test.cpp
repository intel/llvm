// REQUIRES: cuda
// RUN: clang++ -x cuda %s -o %t.out -L/usr/local/cuda/lib64 -lcudart
// RUN: %t.out

#include <cuda.h>

__global__ void test() {
  if (threadIdx.x == 0)
    printf("Hello!\n");
}

int main() {
  test<<<1, 10>>>();
  cudaDeviceSynchronize();
}
