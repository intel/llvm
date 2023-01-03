// Test checks that !sycl-framework metadata is emitted only to functions
// whose topmost namespace is sycl.

// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

// Check that all functions on sycl::* namespace are marked with !sycl-framework metadata
namespace sycl {
  // CHECK: @_ZN4sycl4bar1Ev() {{.*}} !sycl-framework
  void bar1() {}

  namespace V1 {
    // CHECK: @_ZN4sycl2V14bar2Ev() {{.*}} !sycl-framework
    void bar2() {}
  }
}

// Check that V1::sycl::* functions are not marked with !sycl-framework
// metadata since topmost namespace is V1 instead of sycl.
namespace V1 {
  namespace sycl {
    // CHECK-NOT: @_ZN2V14sycl4bar3Ev() {{.*}} !sycl-framework
    void bar3() {}
  }
}

// Check that user code is not marked with !sycl-framework metadata
// CHECK-NOT: @_Z3foov() {{.*}} !sycl-framework
int foo() {
  return 123;
}

// Check that kernel is not marked with !sycl-framework metadata
// CHECK-NOT: spir_kernel {{.*}} !sycl-framework

int main() {
  sycl::kernel_single_task<class kernel>([]() {
    foo();
    sycl::bar1();
    sycl::V1::bar2();
    V1::sycl::bar3();
  });
}
