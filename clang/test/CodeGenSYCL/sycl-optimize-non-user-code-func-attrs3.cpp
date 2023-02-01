// Test checks that noinline and optnone function's attributes are removed and
// are added back correctly depending on presence or absence of alwaysinline
// and minsize function's attributes.

#include "sycl.hpp"

// Check that noinline and optnone attrs are removed from bar1 and bar2. Also
// check that noinline and optnone attrs aren't present in bar3 and bar4
// definitions since they contain minsize and alwaysinline attrs.
// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-BEFORE-OPT

// CHECK-BEFORE-OPT: define {{.*}} @_ZN4sycl4bar1Ev() #[[ATTRS_WITHOUT_NOINLINE_OPTNONE:[0-9]+]]
// CHECK-BEFORE-OPT: define {{.*}} @_ZN4sycl4bar2Ev() #[[ATTRS_WITHOUT_NOINLINE_OPTNONE]]
// CHECK-BEFORE-OPT: define {{.*}} @_ZN4sycl4bar3Ev() #[[ATTRS_WITH_ALWAYSINLINE:[0-9]+]]
// CHECK-BEFORE-OPT: define {{.*}} @_ZN4sycl4bar4Ev() #[[ATTRS_WITH_MINSIZE:[0-9]+]]

// CHECK-BEFORE-OPT: attributes #[[ATTRS_WITH_ALWAYSINLINE]] = {{.*}} alwaysinline
// CHECK-BEFORE-OPT-NOT: attributes #[[ATTRS_WITHOUT_NOINLINE_OPTNONE]] = {{.*}} {{noinline|optnone}}
// CHECK-BEFORE-OPT: attributes #[[ATTRS_WITH_MINSIZE]] = {{.*}} minsize


// Check that noinline and optnone attrs are returned back to bar1 and bar2
// but aren't being added to bar3 and bar4 since they have minsize and
// alwaysinline attrs.
// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -internal-isystem %S/Inputs -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-AFTER-OPT

// CHECK-AFTER-OPT: define {{.*}} @_ZN4sycl4bar1Ev() #[[ATTRS_WITH_NOINLINE_OPTNONE:[0-9]+]]
// CHECK-AFTER-OPT: define {{.*}} @_ZN4sycl4bar2Ev() #[[ATTRS_WITH_NOINLINE_OPTNONE]]
// CHECK-AFTER-OPT: define {{.*}} @_ZN4sycl4bar3Ev() #[[ATTRS_WITH_ALWAYSINLINE:[0-9]+]]
// CHECK-AFTER-OPT: define {{.*}} @_ZN4sycl4bar4Ev() #[[ATTRS_WITH_MINSIZE:[0-9]+]]

// CHECK-AFTER-OPT: attributes #[[ATTRS_WITH_NOINLINE_OPTNONE]] = {{.*}} {{noinline|optnone}} {{.*}} {{noinline|optnone}}
// CHECK-AFTER-OPT: attributes #[[ATTRS_WITH_ALWAYSINLINE]] = {{.*}} alwaysinline
// CHECK-AFTER-OPT: attributes #[[ATTRS_WITH_MINSIZE]] = {{.*}} minsize

namespace sycl {
  void bar1() {}

  void __attribute__((noinline)) bar2() {}

  void __attribute__((always_inline)) bar3() {}

  void __attribute__((minsize)) bar4() {}
}

int main() {
  sycl::kernel_single_task<class kernel>([]() {
    sycl::bar1();
    sycl::bar2();
    sycl::bar3();
    sycl::bar4();
  });
}
