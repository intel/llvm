// REQUIRES: linux
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O2 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_HOST -O2 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-HOST --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_SHARED -O2 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-SHARED --input-file %t.txt %s
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 16;
  constexpr std::size_t M = 16;
  constexpr std::size_t K = N / 4;
#if defined(MALLOC_HOST)
  auto matrixA = (int(*)[N])sycl::malloc_host<int>(N * M, Q);
  auto matrixB = (int(*)[N])sycl::malloc_host<int>(N * M, Q);
  auto matrixC = (int(*)[N])sycl::malloc_host<int>(N * M, Q);
#elif defined(MALLOC_SHARED)
  auto matrixA = (int(*)[N])sycl::malloc_shared<int>(N * M, Q);
  auto matrixB = (int(*)[N])sycl::malloc_shared<int>(N * M, Q);
  auto matrixC = (int(*)[N])sycl::malloc_shared<int>(N * M, Q);
#elif defined(MALLOC_DEVICE)
  auto matrixA = (int(*)[N])sycl::malloc_device<int>(N * M, Q);
  auto matrixB = (int(*)[N])sycl::malloc_device<int>(N * M, Q);
  auto matrixC = (int(*)[N])sycl::malloc_device<int>(N * M, Q);
#elif defined(MALLOC_SYSTEM)
  auto matrixA = (int(*)[N])new int[N * M];
  auto matrixB = (int(*)[N])new int[N * M];
  auto matrixC = (int(*)[N])new int[N * M];
#else
#error "Must provide malloc type to run the test"
#endif

  Q.single_task([=]() {
    for (unsigned m = 0; m < M; ++m) {
      for (unsigned n = 0; n < N; ++n) {
        matrixA[m][n] = n;
        matrixB[m][n] = n + m;
        matrixC[m][n] = 0;
      }
    }
  });
  Q.wait();

  Q.submit([&](sycl::handler &h) {
    // Local accessor, for one matrix tile:
    constexpr unsigned int tile_size = 16;
    local_accessor<int> tileA{tile_size, h};
    h.parallel_for<class MatMultiply>(
        nd_range<2>{{M, N}, {1, tile_size}}, [=](nd_item<2> item) {
          // Indices in the global index space:
          int m = item.get_global_id()[0];
          int n = item.get_global_id()[1];
          // Index in the local index space:
          int i = item.get_local_id()[1];
          int sum = 0;
          for (unsigned int kk = 0; kk < K; kk += tile_size) {
            // Load the matrix tile from matrix A, and synchronize
            // to ensure all work-items have a consistent view
            // of the matrix tile in local memory.
            tileA[i] = matrixA[m][kk + i + 1]; // <== bug add "+1" intentionally
            // CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on USM Device Memory
            // CHECK-HOST:   ERROR: DeviceSanitizer: out-of-bounds-access on USM Host Memory
            // CHECK-SHARED: ERROR: DeviceSanitizer: out-of-bounds-access on USM Shared Memory
            // CHECK: {{READ of size 4 at kernel <.*MatMultiply> LID\(15, 0, 0\) GID\(15, 15, 0\)}}
            // CHECK: {{  #0 .* .*matrix_multiply.cpp:}}[[@LINE-5]]
            item.barrier();
            // Perform computation using the local memory tile, and
            // matrix B in global memory.
            for (unsigned int k = 0; k < tile_size; k++)
              sum += tileA[k] * matrixB[kk + k][n];
            // After computation, synchronize again, to ensure all
            // reads from the local memory tile are complete.
            item.barrier();
          }
          // Write the final result to global memory.
          matrixC[m][n] = sum;
        });
  });
  Q.wait();

  return 0;
}
