// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsycl-explicit-simd -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -verify %s

#include "Inputs/sycl.hpp"

// expected-error@+2 {{'named_sub_group_size' and 'sub_group_size' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sub_group_size(1)]][[intel::named_sub_group_size(automatic)]]
void f1();
// expected-error@+2 {{'sub_group_size' and 'named_sub_group_size' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]][[intel::sub_group_size(1)]]
void f2();

// expected-error@+1 {{'sub_group_size' and 'named_sub_group_size' attributes are not compatible}}
[[intel::sub_group_size(1)]]
void f3();
// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]]
void f3();

// expected-error@+1 {{'named_sub_group_size' and 'sub_group_size' attributes are not compatible}}
[[intel::named_sub_group_size(primary)]]
void f4();
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sub_group_size(1)]]
void f4();

// expected-note@+1 {{previous attribute is here}}
[[intel::named_sub_group_size(automatic)]]
void f5();

// expected-warning@+1 {{attribute 'named_sub_group_size' is already applied with different arguments}}
[[intel::named_sub_group_size(primary)]]
void f5();

[[intel::named_sub_group_size(automatic)]]
void f6();

[[intel::named_sub_group_size(automatic)]]
void f6();

// expected-warning@+1 {{'named_sub_group_size' attribute argument not supported: 'invalid'}}
[[intel::named_sub_group_size(invalid)]]
void f7();

// expected-error@+2 {{'named_sub_group_size' and 'sycl_explicit_simd' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sycl_explicit_simd]][[intel::named_sub_group_size(automatic)]]
void f8();
// expected-error@+2 {{'sub_group_size' and 'sycl_explicit_simd' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sycl_explicit_simd]][[intel::sub_group_size(1)]]
void f9();

// expected-error@+1 {{'named_sub_group_size' and 'sycl_explicit_simd' attributes are not compatible}}
[[intel::named_sub_group_size(primary)]]
void f10();
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sycl_explicit_simd]]
void f10();

void NoAttrFunc(){}
SYCL_EXTERNAL void NoAttrExternalDefined() {}
SYCL_EXTERNAL void NoAttrExternalNotDefined(); // #NoAttrExternalNotDefined

struct Functor {
  [[intel::named_sub_group_size(primary)]] void operator()() const {
    NoAttrFunc();
    NoAttrExternalDefined();
    // expected-error@#NoAttrExternalNotDefined{{undefined 'SYCL_EXTERNAL' function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@-4{{conflicting attribute is here}}
    NoAttrExternalNotDefined();
  }
};

void calls_kernel_1() {
  sycl::kernel_single_task<class Kernel1>([]() [[intel::named_sub_group_size(primary)]] {
    NoAttrFunc();
    NoAttrExternalDefined();
    // expected-error@#NoAttrExternalNotDefined{{undefined 'SYCL_EXTERNAL' function must have a sub group size that matches the size specified for the kernel}}
    // expected-note@-4{{conflicting attribute is here}}
    NoAttrExternalNotDefined();
  });
}

void calls_kernel_2() {
  Functor F;
  sycl::kernel_single_task<class Kernel2>(F);
}

// Func w/o attr called from kernel, kernel has attr.
// normal func: fine
// defined SYCL_EXTERNAL: fine
// undef SYCL_EXTERNAL: Not fine
// all are OK if kernel has 'default' attr.

// Func w attr called from kernel, kernel has attr.
// first matches default.
// kernel matches default
// both matches default
// +SYCL_EXTERNAL

// Func w attr called from kernel, kernel has no attr.
// first matches default.
// kernel matches default
// both matches default
// +SYCL_EXTERNAL
