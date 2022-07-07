// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -verify -pedantic -fsyntax-only -x c++ %s
// RUN: %clang_cc1 -verify -pedantic -fsyntax-only -x c++ %s

#define PARAM_1 1U << 7
#define PARAM_2 1U << 8

// This test makes sure that the compiler checks the semantics of
// __builtin_intel_fpga_mem built-in function arguments correctly.

#ifdef __SYCL_DEVICE_ONLY__
static_assert(__has_builtin(__builtin_intel_fpga_mem), "");
struct State {
  int x;
  float y;
};

struct inner {
  void (*fp)(); // expected-note {{field with illegal type declared here}}
};

struct outer {
  inner A;
};

void foo(float *A, int *B, State *C) {
  float *x;
  int *y;
  State *z;
  int i = 0;
  void *U;

  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, -1);
  // expected-error@-1{{builtin parameter must be a non-negative integer constant}}
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2);
  // expected-error@-1{{too few arguments to function call, expected 3, have 2}}
  y = __builtin_intel_fpga_mem(B, 0, i);
  // expected-error@-1{{argument to '__builtin_intel_fpga_mem' must be a constant integer}}
  z = __builtin_intel_fpga_mem(C, i, 0);
  // expected-error@-1{{argument to '__builtin_intel_fpga_mem' must be a constant integer}}
  z = __builtin_intel_fpga_mem(U, 0, 0);
  // expected-error@-1{{illegal pointer argument of type 'void *'  to __builtin_intel_fpga_mem; only pointers to a first class lvalue or to an rvalue are allowed}}

  int intArr[10] = {0};
  int *k1 = __builtin_intel_fpga_mem(intArr, 0, 0);
  // expected-error@-1{{builtin parameter must be a pointer}}

  int **k2 = __builtin_intel_fpga_mem(&intArr, 0, 0);
  // expected-error@-1{{illegal pointer argument of type 'int (*)[10]'  to __builtin_intel_fpga_mem; only pointers to a first class lvalue or to an rvalue are allowed}}

  void (*fp1)();
  void (*fp2)() = __builtin_intel_fpga_mem(fp1, 0, 0);
  // expected-error@-1{{illegal pointer argument of type 'void (*)()'  to __builtin_intel_fpga_mem; only pointers to a first class lvalue or to an rvalue are allowed}}

  struct outer *iii;
  struct outer *iv = __builtin_intel_fpga_mem(iii, 0, 0);
  // expected-error@-1{{illegal field in type pointed to by pointer argument to __builtin_intel_fpga_mem; only pointers to a first class lvalue or to an rvalue are allowed}}

  // Up to 7 parameters is ok.
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1);
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, 10);
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, 10, 20);
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, -1, 10, 20);

  z = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, B, -1, 1, 1);
  // expected-error@-1{{argument to '__builtin_intel_fpga_mem' must be a constant integer}}
  z = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, U, 10, 20);
  // expected-error@-1{{argument to '__builtin_intel_fpga_mem' must be a constant integer}}
  z = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, -1, C, 300);
  // expected-error@-1{{argument to '__builtin_intel_fpga_mem' must be a constant integer}}
  z = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, -1, 1, i);
  // expected-error@-1{{argument to '__builtin_intel_fpga_mem' must be a constant integer}}

  y = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 1, 1, -1, 10, 20, 30);
  // expected-error@-1{{too many arguments to function call, expected at most 7, have 8}}
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() {
      float *A;
      int *B;
      State *C;
      foo(A, B, C); });
  return 0;
}

#else
void bar(float *A) {
  float *x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 127);
  // expected-error@-1{{'__builtin_intel_fpga_mem' is only available in SYCL device}}
}

int main() {
  float *A;
  bar(A);
  return 0;
}
#endif
