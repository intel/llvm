// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL-ID %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-host -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-host -sycl-std=2020 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -sycl-std=2020 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -sycl-std=1.2.1 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-DEVICE %s
// RUNx: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -E -dM -fms-compatibility | FileCheck --check-prefix=CHECK-MSVC %s
// RUN: %clang_cc1 -fno-sycl-id-queries-fit-in-int %s -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-NO-SYCL_FIT_IN_INT %s

// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-SYCL-BFLOAT16-CONV-DISABLED %s
// RUN: %clang_cc1 %s -fsycl-is-host -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-SYCL-BFLOAT16-CONV-DISABLED %s
// RUN: %clang_cc1 -fsycl-is-device -fsycl-enable-bfloat16-conversion -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-SYCL-BFLOAT16-CONV %s
// RUN: %clang_cc1 -fsycl-is-host -fsycl-enable-bfloat16-conversion -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-SYCL-BFLOAT16-CONV %s
// RUN: %clang_cc1 -fsycl-is-device -fno-sycl-enable-bfloat16-conversion -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-SYCL-BFLOAT16-CONV-DISABLED %s
// RUN: %clang_cc1 -fsycl-is-host -fno-sycl-enable-bfloat16-conversion -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-SYCL-BFLOAT16-CONV-DISABLED %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define SYCL_EXTERNAL
// CHECK-NOT:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-NOT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-STD:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL-STD:#define SYCL_LANGUAGE_VERSION 201707
// CHECK-SYCL-STD:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-STD-2020:#define SYCL_LANGUAGE_VERSION 202001

// CHECK-SYCL-STD-DEVICE:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-SYCL-STD-DEVICE:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-MSVC-NOT: __GNUC__
// CHECK-MSVC-NOT: __STDC__
// CHECK-MSVC: #define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-NO-SYCL_FIT_IN_INT-NOT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1
// CHECK-SYCL-ID:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-BFLOAT16-CONV:#define __SYCL_BF16_CONVERSION_IS_SUPPORTED__ 1
// CHECK-SYCL-BFLOAT16-CONV-DISABLED-NOT:#define __SYCL_BF16_CONVERSION_IS_SUPPORTED__ 1
