// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -verify -pedantic -fsyntax-only -internal-isystem %S/Inputs %s
// RUN: %clang_cc1 -verify -pedantic -fsyntax-only %s

// This test makes sure that the compiler checks the semantics of
// __builtin_intel_sycl_ptr_annotation built-in function arguments correctly.

#ifdef __SYCL_DEVICE_ONLY__
#include "sycl.hpp"
static_assert(__has_builtin(__builtin_intel_sycl_ptr_annotation), "");
struct State {
  int x;
  float y;
};

void foo(float *A, int *B, State *C) {
  float *x;
  int *y;
  State *z;
  int i = 0;
  void *U;
  char* p;

  __builtin_intel_sycl_ptr_annotation();
  // expected-error@-1{{too few arguments to function call, expected 1, have 0}}
  x = __builtin_intel_sycl_ptr_annotation(A, "test");
  // expected-error@-1{{number of parameters must be odd number}}
  y = __builtin_intel_sycl_ptr_annotation(B, "test", i);
  // expected-error@-1{{argument to '__builtin_intel_sycl_ptr_annotation' must be a constant integer}}
  z = __builtin_intel_sycl_ptr_annotation(C, p, 0);
  // expected-error@-1{{builtin parameter must be a string literal or constexpr const char*}}

  int intArr[10] = {0};
  int *k1 = __builtin_intel_sycl_ptr_annotation(intArr, "test", 0);
  // expected-error@-1{{builtin parameter must be a pointer}}

  x = __builtin_intel_sycl_ptr_annotation(A, "A", "B", 1, 1);
  x = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", 1, 1, 10);
  x = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", "D", 1, 1, 10, 20);
  x = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", "D", "E", 1, 1, -1, 10, 20);

  // constexpr can be evaluated
  constexpr const char* str = "abcdefg";
  x = __builtin_intel_sycl_ptr_annotation(A, str, 20);

  z = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", "D", "E", 1, B, -1, 1, 1);
  // expected-error@-1{{argument to '__builtin_intel_sycl_ptr_annotation' must be a constant integer}}
  z = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", "D", "E", 1, 1, U, 1, 1);
  // expected-error@-1{{argument to '__builtin_intel_sycl_ptr_annotation' must be a constant integer}}
  z = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", "D", "E", 1, 1, -1, C, 300);
  // expected-error@-1{{argument to '__builtin_intel_sycl_ptr_annotation' must be a constant integer}}
  z = __builtin_intel_sycl_ptr_annotation(A, "A", "B", "C", "D", "E", 1, 1, -1, 1, i);
  // expected-error@-1{{argument to '__builtin_intel_sycl_ptr_annotation' must be a constant integer}}
}

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class kernel>([=](){
        float *A;
        int *B;
        State *C;
        foo(A, B, C);
    });
  });
  return 0;
}

#else
void bar(float *A) {
  float *x = __builtin_intel_sycl_ptr_annotation(A, "test", 127);
  // expected-error@-1{{'__builtin_intel_sycl_ptr_annotation' is only available in SYCL device}}
}

int main() {
  float *A;
  bar(A);
  return 0;
}
#endif
