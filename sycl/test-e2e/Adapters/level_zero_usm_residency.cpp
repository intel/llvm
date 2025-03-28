// REQUIRES: gpu, level_zero
// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-INTENDED: v2 adapter does not support changing residency.

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_UR_TRACE=2 UR_L0_DEBUG=-1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=DEVICE %s
// RUN: env SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x001 SYCL_UR_TRACE=2 UR_L0_DEBUG=-1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=DEVICE %s
// RUN: env SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x010 SYCL_UR_TRACE=2 UR_L0_DEBUG=-1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=SHARED %s
// RUN: env SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x100 SYCL_UR_TRACE=2 UR_L0_DEBUG=-1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=HOST %s

// Test that USM is made resident at allocation as requested.

#include <sycl/usm.hpp>

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
  queue Q;

  auto ptr1 = malloc_device<int>(1, Q);
  // DEVICE: ---> urUSMDeviceAlloc
  // DEVICE: zeMemAllocDevice
  // DEVICE: zeContextMakeMemoryResident

  auto ptr2 = malloc_shared<int>(1, Q);
  // SHARED: ---> urUSMSharedAlloc
  // SHARED: zeMemAllocShared
  // SHARED: zeContextMakeMemoryResident
  // SHARED-NOT: zeContextMakeMemoryResident

  auto ptr3 = malloc_host<int>(1, Q);
  // HOST: ---> urUSMHostAlloc
  // HOST: zeMemAllocHost
  // HOST: zeContextMakeMemoryResident

  free(ptr1, Q);
  free(ptr2, Q);
  free(ptr3, Q);
  return 0;
}
