// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -verify -pedantic -fsyntax-only %s

void __builtins_AS_3(__attribute__((address_space(3))) int *);

// Check calling a function using address space 3 (local for AMD) pointer works
// with __local.
__kernel void ker(__local int *IL) {
  __builtins_AS_3(IL);
}

// Check casting __local to address space 3 (local for AMD) pointer works.
__kernel void ker_2(__global int *Array, int N) {
  __local int IL;
  __attribute__((address_space(3))) int *I3;
  I3 = (__attribute__((address_space(3))) int *)&IL;
  Array[N] = *I3;
}

// Check casting __shared to address space 5 (private for AMD) pointer errors.
__kernel void ker_3(__global int *Array, int N) {
  __private int IP;
  __attribute__((address_space(5))) int *I5;
  I5 = (__attribute__((address_space(5))) int *)&IP;
  Array[N] = *I5;
}

// Check casting of address_space(3) to __generic pointer works.
__kernel void ker_4(__global int *Array, int N, __attribute__((address_space(3))) int *AS3_ptr) {
  __generic int *IG;
  IG = AS3_ptr; // expected-warning {{assigning to '__generic int *__private' from '__attribute__((address_space(3))) int *__private' discards qualifiers}}
}
