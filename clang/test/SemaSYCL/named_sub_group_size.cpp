// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -sycl-std=2017 -DSYCL2017 -verify %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsycl-default-sub-group-size=primary -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -ast-dump -verify=expected,integer -DSYCL2020 %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsycl-default-sub-group-size=10 -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -ast-dump -verify=expected,primary -DSYCL2020 %s

// Validate the semantic analysis checks for the named_sub_group_size attribute.

#include "Inputs/sycl.hpp"

#if defined(SYCL2017)
// Test that we get ignored attribute warning when using
// a [[intel::named_sub_group_size()]] attribute spelling while not
// in SYCL 2020 mode.
[[intel::named_sub_group_size(automatic)]] void func_ignore(); // expected-warning {{'named_sub_group_size' attribute ignored}}
#endif // SYCL2017

#if defined(SYCL2020)
// The kernel has an attribute.
void calls_kernel_1() {
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ14calls_kernel_1vE7Kernel1
  // CHECK      : IntelNamedSubGroupSizeAttr {{.*}} Automatic
  sycl::kernel_single_task<class Kernel1>([]() [[intel::named_sub_group_size(automatic)]] {
  });
}

struct Functor {
  [[intel::named_sub_group_size(automatic)]] void operator()() const {
  }
};

struct Functor1 {
  [[intel::named_sub_group_size(primary)]] void operator()() const {
  }
};

// Test attributes get propgated to the kernel.
void calls_kernel_2() {
  Functor F;
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ14calls_kernel_2vE7Kernel2
  // CHECK      : IntelNamedSubGroupSizeAttr {{.*}} Automatic
  sycl::kernel_single_task<class Kernel2>(F);

  Functor1 F1;
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ14calls_kernel_3vE7Kernel3
  // CHECK      : IntelNamedSubGroupSizeAttr {{.*}} Primary
  sycl::kernel_single_task<class Kernel3>(F1);
}

// Test ttribute does not get propgated to the kernel.
[[intel::named_sub_group_size(primary)]] void AttrFunc() {} // #AttrFunc

void calls_kernel_3() {
  // CHECK-LABEL: FunctionDecl {{.*}}_ZTSZ14calls_kernel_4vE7Kernel4
  // CHECK-NOT  : IntelNamedSubGroupSizeAttr {{.*}}
  sycl::kernel_single_task<class Kernel4>([]() { // #Kernel4
    // primary-error@#AttrFunc{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // primary-note@#Kernel4{{kernel declared here}}
    AttrFunc();
  });
}

// The kernel has an attribute.
void calls_kernel_4() {
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ14calls_kernel_5vE7Kernel5
  // CHECK      : IntelNamedSubGroupSizeAttr {{.*}} Automatic
  sycl::kernel_single_task<class Kernel5>([]() [[intel::named_sub_group_size(automatic)]] { // #Kernel5
    // expected-error@#AttrFunc{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@#Kernel5{{conflicting attribute is here}}
    AttrFunc();
  });
}
#endif // SYCL2020
