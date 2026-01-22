// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

// This test checks specifics of semantic analysis of ESIMD private globals

// No error expected. SYCL private globals are allowed to have initializers
__attribute__((opencl_private)) int syclPrivGlob;
__attribute__((opencl_private)) int syclPrivGlobInit = 10;

// expected-error@+1{{SYCL explicit SIMD does not permit private global variable to have an initializer}}
__attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) int esimdPrivGlobInit = 10;

// No error expected. ESIMD private globals without initializers are OK
__attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) int esimdPrivGlob;
