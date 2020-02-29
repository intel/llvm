// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL %s
// RUN: %clang_cc1 %s -fsycl -E -dM | FileCheck --check-prefix=CHECK-ANY-SYCL %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM -fms-compatibility | FileCheck --check-prefix=CHECK-MSVC %s
// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define SYCL_EXTERNAL
// CHECK-NOT:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-ANY-SYCL-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-ANY-SYCL:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL:#define SYCL_EXTERNAL __attribute__((sycl_device))
// CHECK-MSVC-NOT: __GNUC__
// CHECK-MSVC-NOT: __STDC__
