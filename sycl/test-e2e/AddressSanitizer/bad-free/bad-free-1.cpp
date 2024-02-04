// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
#include <sycl/sycl.hpp>

constexpr size_t N = 64;

int main() {
  sycl::queue Q;

#if defined(MALLOC_HOST)
  auto *data = sycl::malloc_host<int>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *data = sycl::malloc_shared<int>(N, Q);
#elif defined(MALLOC_DEVICE)
  auto *data = sycl::malloc_device<int>(N, Q);
#else
  auto *data = new int[N];
#endif

#if defined(MALLOC_HOST) || defined(MALLOC_SHARED) || defined(MALLOC_DEVICE)
  sycl::free(data + 1, Q);
#else
  delete[] (data + 1);
#endif
  return 0;
}
