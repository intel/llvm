// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL-ID %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-DEVICE %s
// RUNx: %clang_cc1 %s -fsycl-id-queries-fit-in-int -fsycl-is-device -E -dM -fms-compatibility | FileCheck --check-prefix=CHECK-MSVC %s
// RUN: %clang_cc1 -fno-sycl-id-queries-fit-in-int %s -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-NO-SYCL_FIT_IN_INT %s
// RUN: %clang_cc1 %s  -triple nvptx64-nvidia-cuda -target-cpu sm_80 -fsycl-is-device -E -dM | FileCheck \
// RUN: --check-prefix=CHECK-CUDA %s -DARCH_CODE=800
// RUN: %clang_cc1 %s  -triple amdgcn-amd-amdhsa -target-cpu gfx906 -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-HIP %s

// RUN: %clang_cc1 %s  -triple nvptx64-nvidia-cuda -target-cpu sm_90a -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-CUDA-FEATURE %s
// RUN: %clang_cc1 %s  -triple nvptx64-nvidia-cuda -target-cpu sm_90a -fsycl-is-device -fsycl-cuda-compatibility -E -dM | FileCheck --check-prefix=CHECK-SYCL-CUDA-COMPAT %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define SYCL_EXTERNAL
// CHECK-NOT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-STD:#define SYCL_LANGUAGE_VERSION 202012L

// CHECK-SYCL-STD:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-STD-2020:#define SYCL_LANGUAGE_VERSION 202012
// CHECK-SYCL-STD-2020:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-SYCL-STD-DEVICE:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-SYCL-STD-DEVICE:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-MSVC-NOT: __GNUC__
// CHECK-MSVC-NOT: __STDC__
// CHECK-MSVC: #define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-NO-SYCL_FIT_IN_INT-NOT:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1
// CHECK-SYCL-ID:#define __SYCL_ID_QUERIES_FIT_IN_INT__ 1

// CHECK-CUDA:#define __SYCL_CUDA_ARCH__ [[ARCH_CODE]]
// CHECK-CUDA-NOT:#define __CUDA_ARCH__ {{[0-9]+}}
// CHECK-SYCL-CUDA-COMPAT:#define __CUDA_ARCH__ {{[0-9][0-9]+}}

// CHECK-HIP-NOT:#define __CUDA_ARCH__ {{[0-9]+}}

// CHECK-CUDA-FEATURE:#define __CUDA_ARCH_FEAT_SM90_ALL 1
