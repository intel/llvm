// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s
// XFAIL: *

#include <sycl/types.hpp>

// -------------------------------------------------------------------------

// Not calling constructor, just initing to 0:

// CHECK-LABEL: func.func @vec_default()
// CHECK: sycl.constructor

extern "C" SYCL_EXTERNAL void vec_default() {
  auto vec = sycl::vec<sycl::cl_int, 4>{};
}
