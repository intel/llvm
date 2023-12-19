// REQUIRES: gpu, level_zero

// https://github.com/intel/llvm/issues/11434
// XFAIL: gpu-intel-dg2

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=DEVICE %s
// RUN: env SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x001 SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=DEVICE %s
// RUN: env SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x010 SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=SHARED %s
// RUN: env SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x100 SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=HOST %s

// Test that USM is made resident at allocation as requested.

#include <sycl/sycl.hpp>

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
  queue Q;

  auto ptr1 = malloc_device<int>(1, Q);
  // DEVICE: ---> piextUSMDeviceAlloc
  // DEVICE: ZE ---> zeMemAllocDevice
  // DEVICE: ZE ---> zeContextMakeMemoryResident
  // DEVICE-NOT: ZE ---> zeContextMakeMemoryResident

  auto ptr2 = malloc_shared<int>(1, Q);
  // SHARED: ---> piextUSMSharedAlloc
  // SHARED: ZE ---> zeMemAllocShared
  // SHARED: ZE ---> zeContextMakeMemoryResident
  // SHARED-NOT: ZE ---> zeContextMakeMemoryResident

  auto ptr3 = malloc_host<int>(1, Q);
  // HOST: ---> piextUSMHostAlloc
  // HOST: ZE ---> zeMemAllocHost
  // HOST: ZE ---> zeContextMakeMemoryResident

  free(ptr1, Q);
  free(ptr2, Q);
  free(ptr3, Q);
  return 0;
}
