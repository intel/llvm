// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL-FIT-IN-INT %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-SYCL-FIT-IN-INT %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL-FIT-IN-INT %s
// RUNx: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -E -dM -fms-compatibility | FileCheck --check-prefix=CHECK-MSVC %s
// RUN: %clang_cc1 -fno-sycl-id-queries-fit-in-int %s -E -dM | FileCheck --check-prefix=CHECK-NO-SYCL-FIT-IN-INT %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define SYCL_EXTERNAL
// CHECK-NOT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-FIT-IN-INT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-MSVC-NOT: __GNUC__
// CHECK-MSVC-NOT: __STDC__
// CHECK-MSVC: #define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-NO-SYCL-FIT-IN-INT-NOT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1
