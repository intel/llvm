// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device %s -verify

// Tests __builtin_sycl_launch_kernel: a bare (overloaded / templated) SYCL free
// function kernel name is resolved against the launch argument types via
// ordinary C++ overload resolution / template argument deduction, and the
// builtin evaluates to a pointer to the resolved specialization — usable as the
// `auto *Func` non-type template parameter of the enqueue-function launch path.
// This is the front-end seam the SYCL_EXT_ONEAPI_KERNEL_FUNCTION launch macro is built on.

#include "sycl.hpp"

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void axpy(T *y, const T *x, T a, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = a * x[i] + y[i];
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ovl(int *p) {} // expected-note {{candidate function not viable}}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ovl(float *p) {} // expected-note {{candidate function not viable}}

// A plain function that is NOT a free-function kernel.
__attribute__((sycl_device))
void not_a_kernel(int *p) {} // expected-note {{previous declaration is here}}

// The builtin evaluates to a pointer to the resolved specialization, usable as
// an `auto *Func` non-type template parameter.
template <auto *Func> struct kernel_function_s {};

void test() {
  float *y = nullptr;
  const float *x = nullptr;
  int *ip = nullptr;
  float *fp = nullptr;

  // Deduce T = float from the pointer arguments; result folds to '&axpy<float>'
  // and is usable as a non-type template argument.
  kernel_function_s<__builtin_sycl_launch_kernel(axpy, y, x, 1.0f, 8)> k1;
  (void)k1;

  // Pick the right overload by argument type.
  kernel_function_s<__builtin_sycl_launch_kernel(ovl, ip)> k2;
  (void)k2;
  kernel_function_s<__builtin_sycl_launch_kernel(ovl, fp)> k3;
  (void)k3;

  // No viable overload: no ovl(double*).
  double *dp = nullptr;
  // expected-error@+1 {{no matching function for call to 'ovl'}}
  (void)__builtin_sycl_launch_kernel(ovl, dp);

  // Resolves to a real function that is not a free-function kernel.
  // expected-error@+1 {{is not a SYCL free function kernel}}
  (void)__builtin_sycl_launch_kernel(not_a_kernel, ip);

  // Missing kernel-name argument.
  // expected-error@+1 {{builtin takes one argument}}
  (void)__builtin_sycl_launch_kernel();
}
