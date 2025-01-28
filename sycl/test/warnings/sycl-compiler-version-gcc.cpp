// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ %s -c %t.out | FileCheck %s
// XFAIL: *
// It seems like gcc doesn't properly support _Pragma directive as its own
// documentation says. Therefore, for gcc as host compiler we don't currently
// emit a deprecation warning.
// REQUIRES: linux

#include <sycl/sycl.hpp>

// CHECK: __SYCL_COMPILER_VERSION is deprecated, use __LIBSYCL_TIMESTAMP instead
#if __SYCL_COMPILER_VERSION >= 2024
#endif
