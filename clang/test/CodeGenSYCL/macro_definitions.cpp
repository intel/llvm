// RUN: %clang %s --sycl -sycl-std=1.2.1 -dM -E -x c++ | FileCheck %s
// CHECK: #define CL_SYCL_LANGUAGE_VERSION 121
