// Test checks that noinline and optnone function's attributes aren't attached
// to functions whose topmost namespace is sycl.

// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

// Check that kernel marked with noinline and optnone func attrs.
// CHECK: spir_kernel {{.*}} #[[KERNEL_ATTRS:[0-9]+]]

// Check that user code contain noinline and optnone func attrs.
// CHECK: define {{.*}} @_Z3foov() #[[FOO_ATTRS:[0-9]+]]
int foo() {
  return 123;
}

// Check that all functions on sycl::* namespace do not contain
// noinline and optnone func attrs.
namespace sycl {
  // CHECK: define {{.*}} @_ZN4sycl4bar1Ev() #[[BAR1_ATTRS:[0-9]+]]
  void bar1() {}

  namespace V1 {
    // bar1 and bar2 have common function attrs
    // CHECK: define {{.*}} @_ZN4sycl2V14bar2Ev() #[[BAR1_ATTRS]]
    void bar2() {}
  }
}

// Check that V1::sycl::* functions do not contain noinline and optnone
// func attrs since topmost namespace is V1 instead of sycl.
namespace V1 {
  namespace sycl {
    // foo and bar3 have common function attrs
    // CHECK: define {{.*}} @_ZN2V14sycl4bar3Ev() #[[FOO_ATTRS]]
    void bar3() {}
  }
}

// Check attributes
// CHECK-DAG: attributes #[[KERNEL_ATTRS]] = {{.*}} {{noinline|optnone}} {{.*}} {{noinline|optnone}}
// CHECK-DAG: attributes #[[FOO_ATTRS]] = {{.*}} noinline {{.*}} optnone
// CHECK-NOT: attributes #[[BAR1_ATTRS]] = {{.*}} {{noinline|optnone}}

int main() {
  sycl::kernel_single_task<class kernel>([]() {
    foo();
    sycl::bar1();
    sycl::V1::bar2();
    V1::sycl::bar3();
  });
}
