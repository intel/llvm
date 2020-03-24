// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl -fsycl-is-host -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -sycl-std=1.2.1 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUNx: %clang_cc1 %s -fsycl -fsycl-is-device -E -dM -fms-compatibility | FileCheck --check-prefix=CHECK-MSVC %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define SYCL_EXTERNAL
// CHECK-NOT:#define CL_SYCL_LANGUAGE_VERSION 121

// CHECK-SYCL-STD:#define CL_SYCL_LANGUAGE_VERSION 121

// CHECK-SYCL:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-SYCL:#define SYCL_EXTERNAL __attribute__((sycl_device))

// CHECK-MSVC-NOT: __GNUC__
// CHECK-MSVC-NOT: __STDC__
