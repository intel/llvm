// REQUIRES: gpu, level_zero
// UNSUPPORTED: ze_debug

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// Test that "device_read_only" shared USM allocations are pooled.

#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_shared.hpp>

#include <sycl/usm/usm_allocator.hpp>

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
  queue Q;

  auto ptr1 =
      malloc_shared<int>(1, Q, ext::oneapi::property::usm::device_read_only());
  // CHECK: ---> urUSMSharedAlloc
  // CHECK: zeMemAllocShared

  auto ptr2 = aligned_alloc_shared<int>(
      1, 1, Q, ext::oneapi::property::usm::device_read_only());
  // CHECK: ---> urUSMSharedAlloc
  // CHECK-NOT: zeMemAllocShared

  sycl::usm_allocator<int, sycl::usm::alloc::shared> allocator_no_prop{Q};

  auto ptr3 = allocator_no_prop.allocate(1);
  // CHECK: ---> urUSMSharedAlloc
  // CHECK: zeMemAllocShared

  sycl::usm_allocator<int, sycl::usm::alloc::shared> allocator_prop{
      Q, {sycl::ext::oneapi::property::usm::device_read_only{}}};

  auto ptr4 = allocator_prop.allocate(1);
  // CHECK: ---> urUSMSharedAlloc
  // CHECK-NOT: zeMemAllocShared

  allocator_prop.deallocate(ptr4, 1);
  allocator_no_prop.deallocate(ptr3, 1);
  free(ptr2, Q);
  free(ptr1, Q);
  return 0;
}
