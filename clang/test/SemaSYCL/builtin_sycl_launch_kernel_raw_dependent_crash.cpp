// Boundary / sentinel test for the dependent-context crash.
//
// The SYCL_EXT_ONEAPI_KERNEL_FUNCTION macro wraps the builtin in a unary '+' so that, inside a
// template with dependent arguments, the enclosing kernel_function<...> auto*
// non-type template argument is a UnaryOperator rather than the raw builtin
// CallExpr. Classifying a UnaryOperator does not recurse into
// CallExpr::getCallReturnType, which is what avoids a front-end assertion
// (castAs<FunctionType>() on the builtin's BuiltinFn placeholder callee).
//
// The '+' is a WORKAROUND for a latent front-end defect, not a fix: the RAW
// form (no '+') still crashes. This test pins that boundary. `not --crash`
// asserts clang aborts on the raw form. If a future clang change fixes the
// underlying classification / getCallReturnType path, clang will NO LONGER
// crash here, `not --crash` will fail, and this test will start failing --
// which is the signal to revisit whether the '+' workaround is still needed
// (and whether the underlying fix should be upstreamed).
//
// RUN: not --crash %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device \
// RUN:   -fsyntax-only %s

#include "sycl.hpp"

template <auto *Func> struct kernel_function_s {};
template <auto *Func> constexpr kernel_function_s<Func> kernel_function{};

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void axpy(T *y, const T *x, T a, int n) {}

template <typename T>
auto select(T *y, const T *x, T a, int n) {
  // RAW builtin (no unary '+') as the auto* NTTP, in a dependent context.
  return kernel_function<__builtin_sycl_launch_kernel(axpy, y, x, a, n)>;
}

void use() {
  float *y = nullptr;
  const float *x = nullptr;
  (void)select(y, x, 1.0f, 8);
}
