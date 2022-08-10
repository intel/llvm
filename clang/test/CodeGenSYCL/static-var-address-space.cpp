// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include "Inputs/sycl.hpp"

template <typename T>
void test() {
  // CHECK: @_ZZ4testIiEvvE1a = linkonce_odr addrspace(1) constant i32 0, comdat, align 4
  static const int a = 0;
  // CHECK: @_ZZ4testIiEvvE1b = linkonce_odr addrspace(1) constant i32 0, comdat, align 4
  static const T b = T(0);
}

int main() {
  cl::sycl::kernel_single_task<class fake_kernel>([]() { test<int>(); });
  return 0;
}
