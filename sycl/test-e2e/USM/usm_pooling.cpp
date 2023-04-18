// REQUIRES: level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// Allocate 2 items of 2MB. Free 2. Allocate 3 more of 2MB.

// With no pooling: 1,2,3,4,5 allocs lead to ZE call.
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1 %t.out h 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-NOPOOL
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1 %t.out d 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-NOPOOL
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1 %t.out s 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-NOPOOL

// With pooling enabled and MaxPooolable=1MB: 1,2,3,4,5 allocs lead to ZE call.
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;1M,4,64K" %t.out h 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-12345
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;1M,4,64K" %t.out d 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-12345
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;1M,4,64K" %t.out s 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-12345

// With pooling enabled and capacity=1: 1,2,4,5 allocs lead to ZE call.
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;2M,1,64K" %t.out h 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-1245
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;2M,1,64K" %t.out d 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-1245
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;2M,1,64K" %t.out s 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-1245

// With pooling enabled and MaxPoolSize=2MB: 1,2,4,5 allocs lead to ZE call.
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";2M;2M,4,64K" %t.out h 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-1245
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";2M;2M,4,64K" %t.out d 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-1245
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";2M;2M,4,64K" %t.out s 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-1245

// With pooling enabled and SlabMinSize of 4 MB: 1,5 allocs lead to ZE call.
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;2M,4,4M" %t.out h 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-15
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;2M,4,4M" %t.out d 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-15
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=";;2M,4,4M" %t.out s 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER --check-prefix CHECK-15
#include "CL/sycl.hpp"
#include <iostream>
using namespace sycl;

constexpr size_t SIZE = 2 * 1024 * 1024;

void test_host(context C) {

  void *ph1 = malloc_host(SIZE, C);
  void *ph2 = malloc_host(SIZE, C);
  free(ph1, C);
  free(ph2, C);
  void *ph3 = malloc_host(SIZE, C);
  void *ph4 = malloc_host(SIZE, C);
  void *ph5 = malloc_host(SIZE, C);
  free(ph3, C);
  free(ph4, C);
  free(ph5, C);
}

void test_device(context C, device D) {

  void *ph1 = malloc_device(SIZE, D, C);
  void *ph2 = malloc_device(SIZE, D, C);
  free(ph1, C);
  free(ph2, C);
  void *ph3 = malloc_device(SIZE, D, C);
  void *ph4 = malloc_device(SIZE, D, C);
  void *ph5 = malloc_device(SIZE, D, C);
  free(ph3, C);
  free(ph4, C);
  free(ph5, C);
}

void test_shared(context C, device D) {

  void *ph1 = malloc_shared(SIZE, D, C);
  void *ph2 = malloc_shared(SIZE, D, C);
  free(ph1, C);
  free(ph2, C);
  void *ph3 = malloc_shared(SIZE, D, C);
  void *ph4 = malloc_shared(SIZE, D, C);
  void *ph5 = malloc_shared(SIZE, D, C);
  free(ph3, C);
  free(ph4, C);
  free(ph5, C);
}

int main(int argc, char *argv[]) {
  queue Q;
  device D = Q.get_device();
  context C = Q.get_context();

  const char *devType = D.is_host() ? "Host" : D.is_cpu() ? "CPU" : "GPU";
  std::string pluginName =
      D.get_platform().get_info<sycl::info::platform::name>();
  std::cout << "Running on device " << devType << " ("
            << D.get_info<sycl::info::device::name>() << ") " << pluginName
            << " plugin\n";

  if (*argv[1] == 'h') {
    std::cerr << "Test zeMemAllocHost\n";
    test_host(C);
  } else if (*argv[1] == 'd') {
    std::cerr << "Test zeMemAllocDevice\n";
    test_device(C, D);
  } else if (*argv[1] == 's') {
    std::cerr << "Test zeMemAllocShared\n";
    test_shared(C, D);
  }

  return 0;
}

// CHECK-NOPOOL: Test [[API:zeMemAllocHost|zeMemAllocDevice|zeMemAllocShared]]
// CHECK-NOPOOL-NEXT:  ZE ---> [[API]](
// CHECK-NOPOOL-NEXT:  ZE ---> [[API]](
// CHECK-NOPOOL-NEXT:  ZE ---> zeMemFree
// CHECK-NOPOOL-NEXT:  ZE ---> zeMemFree
// CHECK-NOPOOL-NEXT:  ZE ---> [[API]](
// CHECK-NOPOOL-NEXT:  ZE ---> [[API]](
// CHECK-NOPOOL-NEXT:  ZE ---> [[API]](

// CHECK-12345: Test [[API:zeMemAllocHost|zeMemAllocDevice|zeMemAllocShared]]
// CHECK-12345-NEXT:  ZE ---> [[API]](
// CHECK-12345-NEXT:  ZE ---> [[API]](
// CHECK-12345-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-12345-NEXT:  ZE ---> zeMemFree
// CHECK-12345-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-12345-NEXT:  ZE ---> zeMemFree
// CHECK-12345-NEXT:  ZE ---> [[API]](
// CHECK-12345-NEXT:  ZE ---> [[API]](
// CHECK-12345-NEXT:  ZE ---> [[API]](

// CHECK-1245: Test [[API:zeMemAllocHost|zeMemAllocDevice|zeMemAllocShared]]
// CHECK-1245-NEXT:  ZE ---> [[API]](
// CHECK-1245-NEXT:  ZE ---> [[API]](
// CHECK-1245-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-1245-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-1245-NEXT:  ZE ---> zeMemFree
// CHECK-1245-NEXT:  ZE ---> [[API]](
// CHECK-1245-NEXT:  ZE ---> [[API]](

// CHECK-15: Test [[API:zeMemAllocHost|zeMemAllocDevice|zeMemAllocShared]]
// CHECK-15-NEXT:  ZE ---> [[API]](
// CHECK-15-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-15-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-15-NEXT:  ZE ---> [[API]](
// CHECK-15-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-15-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-15-NEXT:  ZE ---> zeMemGetAllocProperties
// CHECK-15-NEXT:  ZE ---> zeMemFree
