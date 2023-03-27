// REQUIRES: gpu, level_zero

// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

// Test that "device_read_only" shared USM allocations are pooled.

#include <sycl/sycl.hpp>

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
  queue Q;

  auto ptr1 =
      malloc_shared<int>(1, Q, ext::oneapi::property::usm::device_read_only());
  // CHECK: ---> piextUSMSharedAlloc
  // CHECK: ZE ---> zeMemAllocShared

  auto ptr2 = aligned_alloc_shared<int>(
      1, 1, Q, ext::oneapi::property::usm::device_read_only());
  // CHECK: ---> piextUSMSharedAlloc
  // CHECK-NOT: ZE ---> zeMemAllocShared

  free(ptr1, Q);
  free(ptr2, Q);
  return 0;
}
