// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device %s -verify

// Adversarial coverage for __builtin_sycl_launch_kernel name resolution: the
// kernel-name operand must survive the full range of C++ ways a function name
// can be spelled, both non-dependent and (via the '+' form the SYCL_EXT_ONEAPI_KERNEL_FUNCTION
// macro emits) inside templates with dependent arguments.
//
// These lock in the behaviors that were only checked manually before:
//   - qualified names (ns::ft)
//   - explicit template-ids (ns::ft<float>)
//   - overloaded names resolved by argument type, including in a dependent
//     context
//   - mixed template parameters: a leading non-deducible NTTP given explicitly
//     (Dim) plus a trailing deducible type parameter (T)
//   - zero launch arguments

#include "sycl.hpp"

// expected-no-diagnostics

template <auto *Func> struct kernel_function_s {};
template <auto *Func> constexpr kernel_function_s<Func> kernel_function{};

namespace ns {
template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ft(T *p, int n) {}
} // namespace ns

// Overload set, distinguished by argument type.
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ovl(int *p) {}
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ovl(float *p) {}

// Mixed: Dim is a non-deducible NTTP (appears in no parameter), T is deducible.
template <int Dim, typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void mix(T *p, int n) {}

// Zero-argument kernel.
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", "")]]
void noop() {}

// Non-dependent uses.
void non_dependent() {
  float *p = nullptr;
  int *ip = nullptr;

  // Qualified name.
  (void)kernel_function<+__builtin_sycl_launch_kernel(ns::ft, p, 8)>;
  // Qualified explicit template-id.
  (void)kernel_function<+__builtin_sycl_launch_kernel(ns::ft<float>, p, 8)>;
  // Overload resolved by argument type.
  (void)kernel_function<+__builtin_sycl_launch_kernel(ovl, ip)>;
  (void)kernel_function<+__builtin_sycl_launch_kernel(ovl, p)>;
  // Mixed: Dim=2 explicit (leading, non-deducible), T deduced from p.
  (void)kernel_function<+__builtin_sycl_launch_kernel(mix<2>, p, 8)>;
  // Zero launch arguments.
  (void)kernel_function<+__builtin_sycl_launch_kernel(noop)>;
}

// Dependent uses (the '+' form the macro emits): resolution deferred to
// instantiation. These previously risked the front-end classification crash.
template <typename T>
auto dep_qualified(T *p) {
  return kernel_function<+__builtin_sycl_launch_kernel(ns::ft, p, 8)>;
}
template <typename T>
auto dep_overload(T *p) {
  return kernel_function<+__builtin_sycl_launch_kernel(ovl, p)>;
}
template <int Dim, typename T>
auto dep_mixed(T *p) {
  return kernel_function<+__builtin_sycl_launch_kernel(mix<Dim>, p, 8)>;
}

void dependent() {
  int *ip = nullptr;
  float *fp = nullptr;
  (void)dep_qualified(fp);      // ns::ft<float>
  (void)dep_overload(ip);       // ovl(int*)
  (void)dep_overload(fp);       // ovl(float*)
  (void)dep_mixed<3>(fp);       // mix<3, float>
}
