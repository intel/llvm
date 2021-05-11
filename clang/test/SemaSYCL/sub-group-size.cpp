// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -verify=expected,primary,integer %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsycl-default-sub-group-size=primary -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -verify=expected,integer %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsycl-default-sub-group-size=10 -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -verify=expected,primary %s

// Validate the semantic analysis checks for the interaction betwen the
// named_sub_group_size and sub_group_size attributes. These are not able to be
// combined, and require that they only be applied to non-sycl-kernel/
// non-sycl-device functions if they match the kernel they are being called
// from.

#include "Inputs/sycl.hpp"

// expected-error@+2 {{'named_sub_group_size' and 'sub_group_size' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sub_group_size(1)]] [[intel::named_sub_group_size(automatic)]] void f1();
// expected-error@+2 {{'sub_group_size' and 'named_sub_group_size' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]] [[intel::sub_group_size(1)]] void f2();

// expected-note@+1 {{conflicting attribute is here}}
[[intel::sub_group_size(1)]] void f3();
// expected-error@+1 {{'named_sub_group_size' and 'sub_group_size' attributes are not compatible}}
[[intel::named_sub_group_size(primary)]] void f3();

// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]] void f4();
// expected-error@+1 {{'sub_group_size' and 'named_sub_group_size' attributes are not compatible}}
[[intel::sub_group_size(1)]] void f4();

// expected-note@+1 {{previous attribute is here}}
[[intel::named_sub_group_size(automatic)]] void f5();

// expected-warning@+1 {{attribute 'named_sub_group_size' is already applied with different arguments}}
[[intel::named_sub_group_size(primary)]] void f5();

[[intel::named_sub_group_size(automatic)]] void f6();

[[intel::named_sub_group_size(automatic)]] void f6();

// expected-warning@+1 {{'named_sub_group_size' attribute argument not supported: invalid}}
[[intel::named_sub_group_size(invalid)]] void f7();

// expected-error@+2 {{'named_sub_group_size' and 'sycl_explicit_simd' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sycl_explicit_simd]] [[intel::named_sub_group_size(automatic)]] void f8();
// expected-error@+2 {{'sub_group_size' and 'sycl_explicit_simd' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sycl_explicit_simd]] [[intel::sub_group_size(1)]] void f9();

// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]] void f10();
// expected-error@+1 {{'sycl_explicit_simd' and 'named_sub_group_size' attributes are not compatible}}
[[intel::sycl_explicit_simd]] void f10();

// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size("primary")]] void f11();
// expected-error@+1 {{'sycl_explicit_simd' and 'named_sub_group_size' attributes are not compatible}}
[[intel::sycl_explicit_simd]] void f11();

// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size("automatic")]] void f12();
// expected-error@+1 {{'sycl_explicit_simd' and 'named_sub_group_size' attributes are not compatible}}
[[intel::sycl_explicit_simd]] void f12();

// expected-warning@+1 {{'named_sub_group_size' attribute argument not supported: invalid string}}
[[intel::named_sub_group_size("invalid string")]] void f13();

void NoAttrFunc() {}
SYCL_EXTERNAL void NoAttrExternalDefined() {}
SYCL_EXTERNAL void NoAttrExternalNotDefined(); // #NoAttrExternalNotDefined

// If the kernel function has an attribute, only an undefined SYCL_EXTERNAL
// should diagnose.
void calls_kernel_1() {
  sycl::kernel_single_task<class Kernel1>([]() [[intel::named_sub_group_size(automatic)]] {
    NoAttrFunc();
    NoAttrExternalDefined();
    // expected-error@#NoAttrExternalNotDefined{{undefined 'SYCL_EXTERNAL' function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@-4{{conflicting attribute is here}}
    NoAttrExternalNotDefined();
  });
}

struct Functor {
  [[intel::named_sub_group_size(automatic)]] void operator()() const {
    NoAttrFunc();
    //   NoAttrExternalDefined();
    // expected-error@#NoAttrExternalNotDefined{{undefined 'SYCL_EXTERNAL' function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@-4{{conflicting attribute is here}}
    NoAttrExternalNotDefined();
  }
};

void calls_kernel_2() {
  Functor F;
  sycl::kernel_single_task<class Kernel2>(F);
}

// If the kernel doesn't have an attribute,
[[intel::named_sub_group_size(primary)]] void AttrFunc() {}                           // #AttrFunc
[[intel::named_sub_group_size(primary)]] SYCL_EXTERNAL void AttrExternalDefined() {}  // #AttrExternalDefined
[[intel::named_sub_group_size(primary)]] SYCL_EXTERNAL void AttrExternalNotDefined(); // #AttrExternalNotDefined

void calls_kernel_3() {
  sycl::kernel_single_task<class Kernel3>([]() { // #Kernel3
    // primary-error@#AttrFunc{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // primary-note@#Kernel3{{kernel declared here}}
    AttrFunc();
    // primary-error@#AttrExternalDefined{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // primary-note@#Kernel3{{kernel declared here}}
    AttrExternalDefined();
    // primary-error@#AttrExternalNotDefined{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // primary-note@#Kernel3{{kernel declared here}}
    AttrExternalNotDefined();
  });
}

[[intel::sub_group_size(10)]] void AttrFunc2() {}                           // #AttrFunc2
[[intel::sub_group_size(10)]] SYCL_EXTERNAL void AttrExternalDefined2() {}  // #AttrExternalDefined2
[[intel::sub_group_size(10)]] SYCL_EXTERNAL void AttrExternalNotDefined2(); // #AttrExternalNotDefined2

void calls_kernel_4() {
  sycl::kernel_single_task<class Kernel4>([]() { // #Kernel4
    // integer-error@#AttrFunc2{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // integer-note@#Kernel4{{kernel declared here}}
    AttrFunc2();
    // integer-error@#AttrExternalDefined2{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // integer-note@#Kernel4{{kernel declared here}}
    AttrExternalDefined2();
    // integer-error@#AttrExternalNotDefined2{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // integer-note@#Kernel4{{kernel declared here}}
    AttrExternalNotDefined2();
  });
}

// Both have an attribute.
void calls_kernel_5() {
  sycl::kernel_single_task<class Kernel5>([]() [[intel::named_sub_group_size(automatic)]] { // #Kernel5
    // expected-error@#AttrFunc{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@#Kernel5{{conflicting attribute is here}}
    AttrFunc();
    // expected-error@#AttrExternalDefined{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@#Kernel5{{conflicting attribute is here}}
    AttrExternalDefined();
    // expected-error@#AttrExternalNotDefined{{kernel-called function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@#Kernel5{{conflicting attribute is here}}
    AttrExternalNotDefined();

  });
}

// Don't diag with the old sub-group-size.
void calls_kernel_6() {
  sycl::kernel_single_task<class Kernel6>([]() [[intel::reqd_sub_group_size(10)]] { // #Kernel6
    NoAttrExternalNotDefined();
  });
}
