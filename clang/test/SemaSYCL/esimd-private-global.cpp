// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-explicit-simd -fsyntax-only -verify -pedantic %s

// no error expected
__attribute__((opencl_private)) __attribute__((register_num(17))) int privGlob;

// expected-error@+1{{'register_num' attribute takes one argument}}
__attribute__((opencl_private)) __attribute__((register_num())) int privGlob1;

// expected-error@+1{{'register_num' attribute takes one argument}}
__attribute__((opencl_private)) __attribute__((register_num(10, 11))) int privGlob2;

// expected-error@+1{{(SYCL explicit SIMD) private global variable cannot have an initializer}}
__attribute__((opencl_private)) int privGlob3 = 10;

void foo() {
  // expected-warning@+1{{'register_num' attribute only applies to global variables}}
  __attribute__((register_num(17))) int privLoc;
}
