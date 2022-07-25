// RUN: %clangxx -fsycl %s -S -emit-llvm -o- | FileCheck %s
// CHECK-NOT: @_ZStL8__ioinit = internal global %"class.std::ios_base::Init"
//
// Tests if <sycl/sycl.hpp> headers include any <iostream> headers

#include <sycl/sycl.hpp>
using namespace sycl;

int main() { return 0; }
