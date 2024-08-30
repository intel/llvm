// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -triple spir64 \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -Wno-return-type -verify     \
// RUN:  -fsyntax-only -std=c++17 %s

// add_ir_attributes_function attribute used to represent compile-time SYCL
// properties and some of those properties are intended to be turned into
// attributes to enable various diagnostics.
//
// "indirectly-callable" property is supposed to be turned into sycl_device
// attribute to make sure that functions market with that property are being
// diagnosed for violating SYCL device code restrictions.
//
// This test ensures that this is indeed the case.

namespace std {
class type_info; // needed to make typeid compile without corresponding include
} // namespace std

[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void cannot_use_recursion() {
  // expected-error@+2 {{SYCL kernel cannot call a recursive function}}
  // expected-note@-2 {{function implemented using recursion declared here}}
  cannot_use_recursion();
}

[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void cannot_allocate_storage() {
  new int; // expected-error {{SYCL kernel cannot allocate storage}}
}

[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void cannot_use_rtti() {
 (void)typeid(int); // expected-error {{SYCL kernel cannot use rtti}}
}

[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void cannot_use_zero_length_array() {
  // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
  int mosterArr[0];
}

[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void cannot_use_long_double() {
  // expected-error@+1 {{'long double' is not supported on this target}}
  long double terrorLD;
}

[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void cannot_use_exceptions() {
  try { // expected-error {{SYCL kernel cannot use exceptions}}
    ;
  } catch (...) {
    ;
  }
  throw 20; // expected-error {{SYCL kernel cannot use exceptions}}
}
