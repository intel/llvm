/* Test to check that sycl-ls is outputting UUID and number of sub and sub-sub
 * devices. */
// REQUIRES:  gpu, level_zero

// XFAIL: linux && run-mode && arch-intel_gpu_bmg_g21
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/18576

// As of now, ZEX_NUMBER_OF_CCS is not working with FLAT hierachy,
// which is the new default on PVC.

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="level_zero:*" env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE env ZEX_NUMBER_OF_CCS=0:4 sycl-ls --verbose | \
// RUN: FileCheck %s

// CHECK: {{.*}}UUID              : {{.*}}
// CHECK: {{.*}}Num SubDevices    : {{.*}}
// CHECK-NEXT: {{.*}}Num SubSubDevices : {{.*}}
