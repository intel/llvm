// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q1{sycl::gpu_selector_v}, Q2{sycl::cpu_selector_v};
  auto *data = sycl::malloc_device<char>(64, Q1);
  sycl::free(data, Q2);
  return 0;
}
