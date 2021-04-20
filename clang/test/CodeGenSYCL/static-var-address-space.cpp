// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include "Inputs/sycl.hpp"
struct C {
  static int c;
};

template <typename T>
struct D {
  static T d;
};

template <typename T>
void test() {
  // CHECK: @_ZZ4testIiEvvE1a = linkonce_odr addrspace(1) constant i32 0, comdat, align 4
  static const int a = 0;
  // CHECK: @_ZZ4testIiEvvE1b = linkonce_odr addrspace(1) constant i32 0, comdat, align 4
  static const T b = T(0);
  // CHECK: @_ZN1C1cE = external addrspace(1) global i32, align 4
  C::c = 10;
  const C struct_c;
  // CHECK: @_ZN1DIiE1dE = external addrspace(1) global i32, align 4
  D<int>::d = 11;
  const D<int> struct_d;
}

int main() {
  cl::sycl::kernel_single_task<class fake_kernel>([]() { test<int>(); });
  return 0;
}
