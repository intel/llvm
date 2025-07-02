// Test to verify that a warning is thrown when the -fsycl flag is not used and
// <sycl/sycl.hpp> file is included.
// RUN: %clangxx -I %sycl_include -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING
// RUN: %clangxx -I %sycl_include -fsyntax-only -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING %s 2>&1 | FileCheck %s --implicit-check-not=CHECK-WARNING

// CHECK-WARNING: You are including <sycl/sycl.hpp> without -fsycl flag, which is errorenous for device code compilation.
#include <sycl/sycl.hpp>
