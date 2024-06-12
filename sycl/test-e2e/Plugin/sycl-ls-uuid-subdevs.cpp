/* Test to check that sycl-ls is outputting UUID and number of sub and sub-sub
 * devices. */
// REQUIRES:  gpu, level_zero

// UNSUPPORTED: gpu-intel-pvc-1T

// As of now, ZEX_NUMBER_OF_CCS is not working with FLAT hierachy,
// which is the new default on PVC.

// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE env ZEX_NUMBER_OF_CCS=0:4 sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --input-file %t.default.out

// CHECK: {{.*}}UUID              : {{.*}}
// CHECK: {{.*}}Num SubDevices    : {{.*}}
// CHECK-NEXT: {{.*}}Num SubSubDevices : {{.*}}
