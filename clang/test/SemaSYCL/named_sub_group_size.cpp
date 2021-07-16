// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -sycl-std=2020 -fsyntax-only -ast-dump -verify %s | FileCheck %s

// Validate the semantic analysis checks for the named_sub_group_size attribute in SYCL 2020 mode.

#include "Inputs/sycl.hpp"

struct Functor {
  [[intel::named_sub_group_size(automatic)]] void operator()() const {
  }
};

struct Functor1 {
  [[intel::named_sub_group_size(primary)]] void operator()() const {
  }
};

// Test attribute gets propagated to the kernel.
void calls_kernel_1() {
  // CHECK: FunctionDecl {{.*}}Kernel1
  // CHECK: IntelNamedSubGroupSizeAttr {{.*}} Automatic
  sycl::kernel_single_task<class Kernel1>([]() [[intel::named_sub_group_size(automatic)]] {
  });
}

void calls_kernel_2() {
  Functor F;
  // CHECK: FunctionDecl {{.*}}Kernel2
  // CHECK: IntelNamedSubGroupSizeAttr {{.*}} Automatic
  sycl::kernel_single_task<class Kernel2>(F);

  Functor1 F1;
  // CHECK: FunctionDecl {{.*}}Kernel3
  // CHECK: IntelNamedSubGroupSizeAttr {{.*}} Primary
  sycl::kernel_single_task<class Kernel3>(F1);
}

[[intel::named_sub_group_size(primary)]] void AttrFunc() {} // expected-error{{kernel-called function must have a sub group size that matches the size specified for the kernel}}

// Test attribute does not get propagated to the kernel.
void calls_kernel_3() {
  // CHECK:     FunctionDecl {{.*}}Kernel4
  // CHECK-NOT: IntelNamedSubGroupSizeAttr {{.*}}
  sycl::kernel_single_task<class Kernel4>([]() { // expected-note{{kernel declared here}}
    AttrFunc();
  });
}
