// REQUIRES: gpu, level_zero
// UNSUPPORTED: ze_debug

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// Test that "device_read_only" shared USM allocations are pooled.

#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_shared.hpp>

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
  queue Q;

  auto ptr1 =
      malloc_shared<int>(1, Q, ext::oneapi::property::usm::device_read_only());
  // CHECK: ---> urUSMSharedAlloc
  // CHECK-SAME:ZE ---> zeMemAllocShared

  auto ptr2 = aligned_alloc_shared<int>(
      1, 1, Q, ext::oneapi::property::usm::device_read_only());
  // CHECK: ---> urUSMSharedAlloc
  // CHECK-NOT: ZE ---> zeMemAllocShared

  free(ptr1, Q);
  free(ptr2, Q);
  return 0;
}
