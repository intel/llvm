// This test is written with an aim to check that experimental::printf versions
// for constant and generic address space can be used in the same module.
//
// UNSUPPORTED: hip_amd
//
// FIXME: Drop the test once generic AS support is considered stable and the
//        dedicated constant AS overload of printf is removed from the library.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
// CHECK: Constant addrspace literal
// CHECK: Generic addrspace literal

#include <CL/sycl.hpp>

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
