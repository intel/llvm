// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ %s -c %t.out | FileCheck %s
// XFAIL: *
// XFAIL-TRACKER: TBD
// REQUIRES: linux

#include <sycl/sycl.hpp>

// CHECK: __SYCL_COMPILER_VERSION is deprecated, use __LIBSYCL_TIMESTAMP instead
#if __SYCL_COMPILER_VERSION >= 2024

#endif

int main() {}
