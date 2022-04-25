// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -verify -pedantic -fsyntax-only %s

void test_numeric_as_to_generic_implicit_cast(__attribute__((address_space(3))) int *as3_ptr, float src) {
  generic int *generic_ptr = as3_ptr;
}

// AS 4 is constant on AMDGPU, casting it to generic is illegal.
void test_numeric_as_const_to_generic_implicit_cast(__attribute__((address_space(4))) int *as4_ptr, float src) {
  generic int *generic_ptr = as4_ptr; // expected-error{{initializing '__generic int *__private' with an expression of type '__attribute__((address_space(4))) int *__private' changes address space of pointer}}
}

void test_numeric_as_to_generic_explicit_cast(__attribute__((address_space(3))) int *as3_ptr, float src) {
  generic int *generic_ptr = (generic int *)as3_ptr;
}

void test_generic_to_numeric_as_implicit_cast(void) {
  generic int* generic_ptr = 0;
  __attribute__((address_space(3))) int *as3_ptr = generic_ptr; // expected-error{{initializing '__attribute__((address_space(3))) int *__private' with an expression of type '__generic int *__private' changes address space of pointer}}
}

void test_generic_to_numeric_as_explicit_cast(void) {
  generic int* generic_ptr = 0;
  __attribute__((address_space(3))) int *as3_ptr = (__attribute__((address_space(3))) int *)generic_ptr;
}

void test_generic_as_to_builtin_parameter_explicit_cast_numeric(__attribute__((address_space(3))) int *as3_ptr, float src) {
  generic int *generic_ptr = as3_ptr;
  // This is legal, as address_space(3) corresponds to local on amdgpu.
  volatile float result = __builtin_amdgcn_ds_fmaxf((__attribute__((address_space(3))) float *)generic_ptr, src, 0, 0, false);
}

void test_generic_as_to_builtin_parameterimplicit_cast_numeric(__attribute__((address_space(3))) int *as3_ptr, float src) {
  generic int *generic_ptr = as3_ptr;
  volatile float result = __builtin_amdgcn_ds_fmaxf(generic_ptr, src, 0, 0, false); // expected-error {{passing '__generic int *__private' to parameter of type '__local float *' changes address space of pointer}}
}
