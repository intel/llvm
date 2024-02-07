// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
#include <sycl/sycl.hpp>

constexpr size_t N = 64;

int main() {
  sycl::queue Q;
  auto *data = new int[N];
  sycl::free(data, Q);
  return 0;
}
