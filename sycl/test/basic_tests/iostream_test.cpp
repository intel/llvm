// RUN: %clangxx -fsycl -g %s -o %t.out
// RUN: gdb %t.out -batch -x %sycl_gdb_iostream 2>&1 | FileCheck %s
// CHECK: ignore next 9999 hits
//
// Tests if <sycl/sycl.hpp> headers include any <iostream> headers
// Acceptable value is 1.
// Thus the value to check for is 10000-1= 9999

#include <sycl/sycl.hpp>
using namespace sycl;

int main() { return 0; }