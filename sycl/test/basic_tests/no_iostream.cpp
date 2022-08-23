// RUN: %clangxx -fsycl %s -S -emit-llvm -o- | FileCheck %s
// CHECK-NOT: {{^@llvm.global_ctors}}
//
// Tests if inclusion <sycl/sycl.hpp> causes execution of global ctors
// and related performance hit.

#include <sycl/sycl.hpp>
