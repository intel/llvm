// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device %s -verify

// Regression test: __builtin_sycl_launch_kernel used inside a template, with
// dependent launch arguments, must be deferred to instantiation rather than
// resolved eagerly. Using the raw builtin call as the `auto *Func` non-type
// template argument of kernel_function<...> would crash the front end while
// dependent: deducing the auto* parameter runs Expr::Classify on the builtin
// CallExpr, whose callee has the BuiltinFn placeholder type, and
// CallExpr::getCallReturnType asserts trying to castAs<FunctionType>.
//
// The SYCL_EXT_ONEAPI_KERNEL_FUNCTION macro wraps the builtin in a unary plus, so the non-type
// template argument is a UnaryOperator (whose classification is a plain
// prvalue and does not recurse into the dependent CallExpr) instead of the
// CallExpr itself. That is what makes the dependent case well-formed with no
// front-end change; this test spells the same `+` form the macro emits.
// (Unary plus on a function pointer is the identity, so the deduced Func value
// is unchanged.)

#include "sycl.hpp"

template <auto *Func> struct kernel_function_s {};
template <auto *Func> constexpr kernel_function_s<Func> kernel_function{};

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void axpy(T *y, const T *x, T a, int n) {}

// Dependent-argument use inside a function template: deferred to instantiation.
template <typename T>
auto select(T *y, const T *x, T a, int n) {
  return kernel_function<+__builtin_sycl_launch_kernel(axpy, y, x, a, n)>;
}

void ok() {
  // expected-no-diagnostics
  float *y = nullptr;
  const float *x = nullptr;
  (void)select(y, x, 1.0f, 8); // instantiates select<float>, resolves axpy<float>
}
