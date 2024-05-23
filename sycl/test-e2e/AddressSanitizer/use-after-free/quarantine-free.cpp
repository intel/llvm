// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: %force_device_asan_rt UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:5 UR_LOG_SANITIZER=level:info %{run} %t 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

/// Quarantine Cache Test
///
/// The "sycl::free"d buffer are not freed immediately, but enqueued into
/// quarantine cache.
/// The maximum size of quarantine cache (per device) is configured by
/// "quarantine_size_mb" on env "UR_LAYER_ASAN_OPTIONS".
/// If the total size of enqueued buffers is larger than "quarantine_size_mb",
/// then the enqueued buffers will be freed by FIFO.
///
/// In this test, the maximum size of quarantine cache is 5MB (5242880 bytes).

constexpr size_t N = 1024 * 1024;

int main() {
  sycl::queue Q;
  auto *array =
      sycl::malloc_device<char>(N, Q); // allocated size: 1052672 <= 5242880
  // 1. allocated size: {currently the size of all allocated memory} <= {maximum
  // size of quarantine cache}"
  // 2. 1052672 = 1024*1024 + 4096, 4096 is the size of red zone
  // CHECK: Alloc={{\[}}[[ADDR1:0x[0-9a-f]+]]
  sycl::free(array, Q);

  auto *temp =
      sycl::malloc_device<char>(N, Q); // allocated size: 1052672*2 <= 5242880
  // CHECK: Alloc={{\[}}[[ADDR2:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  temp =
      sycl::malloc_device<char>(N, Q); // allocated size: 1052672*3 <= 5242880
  // CHECK: Alloc={{\[}}[[ADDR3:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  temp =
      sycl::malloc_device<char>(N, Q); // allocated size: 1052672*4 <= 5242880
  // CHECK: Alloc={{\[}}[[ADDR4:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  temp = sycl::malloc_device<char>(N, Q); // allocated size: 1052672*5 > 5242880
  // CHECK: Alloc={{\[}}[[ADDR5:0x[0-9a-f]+]]
  sycl::free(temp, Q);
  // CHECK: Quarantine Free: [[ADDR1]]

  return 0;
}
