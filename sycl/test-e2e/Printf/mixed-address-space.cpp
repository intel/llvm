// This test is written with an aim to check that experimental::printf versions
// for constant and generic address space can be used in the same module.
//
// UNSUPPORTED: hip_amd
// XFAIL: cuda && windows
//
// FIXME: Drop the test once generic AS support is considered stable and the
//        dedicated constant AS overload of printf is removed from the library.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// UNSUPPORTED: gpu
// CHECK: Constant addrspace literal
// CHECK: Generic addrspace literal

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

#include "helper.hpp"

#include <cstring>

using namespace sycl;

void test() {
  {
    FORMAT_STRING(constant_literal) = "Constant addrspace literal\n";
    ext::oneapi::experimental::printf(constant_literal);
    ext::oneapi::experimental::printf("Generic addrspace literal\n");
  }
}

class MixedAddrspaceTest;

int main() {
  queue q;

  q.submit([](handler &cgh) {
    cgh.single_task<MixedAddrspaceTest>([]() { test(); });
  });
  q.wait();

  return 0;
}
