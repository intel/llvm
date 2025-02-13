/* Test to check that sycl-ls is outputting UUID and number of sub and sub-sub
 * devices. */
// REQUIRES:  gpu, level_zero

// As of now, ZEX_NUMBER_OF_CCS is not working with FLAT hierachy,
// which is the new default on PVC.

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="level_zero:*" env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE env ZEX_NUMBER_OF_CCS=0:4 sycl-ls --verbose | \
// RUN: FileCheck %s

// CHECK: {{.*}}UUID              : {{.*}}
// CHECK: {{.*}}Num SubDevices    : {{.*}}
// CHECK-NEXT: {{.*}}Num SubSubDevices : {{.*}}
