// REQUIRES: linux && !aspect-ext_oneapi_native_assert
//
// OpenCL GPU drivers do not currently report cl_intel_devicelib_assert despite
// supporting assertions.
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-11496
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// CHECK-NOT: One shouldn't see this message
// CHECK-NOT: {{.*}}assert_in_kernels.hpp:27: void kernelFunc2(int *, int): {{.*}} [{{[0,2]}},0,0], {{.*}} [0,0,0]
// CHECK-NOT: Assertion `Buf[wiID] == 0 && "from assert statement"` failed
// CHECK-NOT: test aborts earlier, one shouldn't see this message
// CHECK: The test ended.
//
// This test checks that devices that do not support native asserts will simply
// skip assertions.

#include "assert_in_kernels.hpp"
