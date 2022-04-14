// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -verify -pedantic -fsyntax-only %s

// The same as address_space_type_cast_amdgpu.cl, but as x86 does not provide
// ASMap all cases should error out.

void __builtins_AS_3(__attribute__((address_space(3))) int *); // expected-note {{passing argument to parameter here}}

// No relatioship between address_space(3) and __local on x86.
__kernel void ker(__local int *IL) {
  __builtins_AS_3(IL); // expected-error {{passing '__local int *__private' to parameter of type '__attribute__((address_space(3))) int *' changes address space of pointer}}
}

// No relatioship between address_space(3) and __local on x86.
__kernel void ker_2(__global int *Array, int N) {
  __local int IL;
  __attribute__((address_space(3))) int *I3;
  I3 = (__attribute__((address_space(3))) int *)&IL; // expected-error {{casting '__local int *' to type '__attribute__((address_space(3))) int *' changes address space of pointer}}
  Array[N] = *I3;
}

// No relatioship between address_space(5) and __private on x86.
__kernel void ker_3(__global int *Array, int N) {
  __private int IP;
  __attribute__((address_space(5))) int *I5;
  I5 = (__attribute__((address_space(5))) int *)&IP; // expected-error {{casting '__private int *' to type '__attribute__((address_space(5))) int *' changes address space of pointer}}
  Array[N] = *I5;
}

// Without ASMap compiler can't tell if address_space(3) is not equal to __constant, fail.
__kernel void ker_4(__global int *Array, int N, __attribute__((address_space(3))) int *AS3_ptr) {
  __generic int *IG;
  IG = AS3_ptr; // expected-error {{assigning '__attribute__((address_space(3))) int *__private' to '__generic int *__private' changes address space of pointer}}
}
