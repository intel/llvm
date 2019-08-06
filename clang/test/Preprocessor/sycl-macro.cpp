// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL %s
// RUN: %clang_cc1 %s -fsycl -E -dM | FileCheck --check-prefix=CHECK-ANY-SYCL %s
// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-ANY-SYCL-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-ANY-SYCL:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL:#define CL_SYCL_LANGUAGE_VERSION 121
