// RUN: %clangxx -fsycl %s -S -emit-llvm -o- | FileCheck %s
// CHECK-NOT: {{^@}}
//
// Tests if <sycl/sycl.hpp> headers include any <iostream> headers

#include <sycl/sycl.hpp>
