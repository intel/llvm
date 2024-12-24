// REQUIRES: linux, cpu || (gpu && level_zero)
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

constexpr size_t _1MB = 1024 * 1024;

int main() {
  sycl::queue Q;

  // allocated size: 1MB+4KB(red zone) <= 5MB
  auto *array = sycl::malloc_device<char>(_1MB, Q);
  // CHECK: Alloc={{\[}}[[ADDR1:0x[0-9a-f]+]]
  sycl::free(array, Q);

  // allocated size: 2MB+8KB < 5MB
  auto *temp = sycl::malloc_device<char>(_1MB, Q);
  // CHECK: Alloc={{\[}}[[ADDR2:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  // allocated size: 3MB+12KB < 5MB
  temp = sycl::malloc_device<char>(_1MB, Q);
  // CHECK: Alloc={{\[}}[[ADDR3:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  // allocated size: 4MB+16KB < 5MB
  temp = sycl::malloc_device<char>(_1MB, Q);
  // CHECK: Alloc={{\[}}[[ADDR4:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  // allocated size: 5MB+20KB > 5MB, old allocation get actual freed
  temp = sycl::malloc_device<char>(_1MB, Q);
  // CHECK: Alloc={{\[}}[[ADDR5:0x[0-9a-f]+]]
  sycl::free(temp, Q);
  // CHECK: Quarantine Free: [[ADDR1]]
  // After free, it becomes 4MB+16KB again

  temp = sycl::malloc_device<char>(_1MB, Q);
  // CHECK: Alloc={{\[}}[[ADDR6:0x[0-9a-f]+]]
  sycl::free(temp, Q);
  // CHECK: Quarantine Free: [[ADDR2]]

  // CHECK-DAG: Quarantine Free: [[ADDR3]]
  // CHECK-DAG: Quarantine Free: [[ADDR4]]
  // CHECK-DAG: Quarantine Free: [[ADDR5]]
  // CHECK-DAG: Quarantine Free: [[ADDR6]]

  return 0;
}
