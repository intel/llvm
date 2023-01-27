// Test checks that noinline and optnone function's attributes aren't attached
// to functions whose topmost namespace is sycl.

// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -internal-isystem %S/Inputs -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

// Check that kernel marked with noinline and optnone func attrs.
// CHECK: spir_kernel {{.*}} #0

// Check that all functions on sycl::* namespace do not contain
// noinline and optnone func attrs.
namespace sycl {
  // CHECK: @_ZN4sycl4bar1Ev() #2
  void bar1() {}

  namespace V1 {
    // CHECK: @_ZN4sycl2V14bar2Ev() #2
    void bar2() {}
  }
}

// Check that V1::sycl::* functions do not contain noinline and optnone
// func attrs since topmost namespace is V1 instead of sycl.
namespace V1 {
  namespace sycl {
    // CHECK-NOT: @_ZN2V14sycl4bar3Ev() #2
    void bar3() {}
  }
}

// Check that user code does not noinline and optnone func attrs.
// CHECK-NOT: @_Z3foov() #1
int foo() {
  return 123;
}

// #0 and #1 contain noinline and optnone func attrs, #2 does not contain them.
// CHECK: attributes #0 = {{.*}} noinline {{.*}} optnone
// CHECK: attributes #1 = {{.*}} noinline {{.*}} optnone
// CHECK-NOT: attributes #2 = {{.*}} {{noinline|optnone}}

int main() {
  sycl::kernel_single_task<class kernel>([]() {
    foo();
    sycl::bar1();
    sycl::V1::bar2();
    V1::sycl::bar3();
  });
}
