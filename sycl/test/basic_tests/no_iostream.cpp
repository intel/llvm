// RUN: %clangxx -fsycl %s -S -emit-llvm -o- | FileCheck %s
// The only global ctor is the one referencing __sycl_registerlib_<id>.
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__sycl_registerlib_ctor, ptr null }]
//
// Tests if inclusion <sycl/sycl.hpp> causes execution of global ctors
// and related performance hit.

#include <sycl/sycl.hpp>
