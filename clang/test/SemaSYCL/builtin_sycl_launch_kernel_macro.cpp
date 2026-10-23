// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsyntax-only %s -verify

// SYCL_EXT_ONEAPI_KERNEL_FUNCTION(name, args...) resolves the kernel as the
// call name(args...) would and launches the selected free function kernel.
// The builtin receives that call already resolved, so these cases check that
// it launches the callee the call selected, not the first or any declaration
// of the name, for each way a call selects its callee: template argument
// deduction, overload resolution, explicit template arguments and
// argument-dependent lookup, in non-dependent and dependent contexts.

#include "sycl_kernel_function_macro.hpp"

#define ST_KERNEL                                                              \
  [[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",    \
                                                "")]]                          \
  __attribute__((sycl_device))

// The kernels of the spec's "templates or overloaded" example.
template <typename T> ND_KERNEL void iota(T start, T *ptr) {}
ST_KERNEL void ping(float *x) {}
ST_KERNEL void ping(int *x) {}

template <int Dim, typename T> ND_KERNEL void fill(T *p, T v) {} // #fill

namespace ns {
struct Params {};
template <int Dim, int Width, typename T> ND_KERNEL void tile(T *p) {}
// Found only through ADL on ns::Params.
ND_KERNEL void only_adl(Params *p) {}
} // namespace ns

void positive(float *fp, int *ip, ns::Params *pp) {
  // Template argument deduction: launches the deduced specialization.
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(iota, 3.14f, fp)), &iota<float>));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(iota, 3, ip)), &iota<int>));
  // Overload resolution: launches the overload selected by the argument types.
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ping, fp)), static_cast<void (*)(float *)>(&ping)));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ping, ip)), static_cast<void (*)(int *)>(&ping)));
  // Explicit template arguments: a non-deducible Dim is given explicitly, and
  // T is still deduced. A template argument list containing a comma must be
  // parenthesized so the preprocessor does not split it.
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill<1>, fp, 1.0f)), &fill<1, float>));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION((ns::tile<1, 16>), fp)), &ns::tile<1, 16, float>));
  // Argument-dependent lookup: only_adl is not visible to ordinary lookup.
  // ADL applies because the macro passes the builtin a call, name(args...);
  // a builtin taking the name and the arguments separately would not find it.
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(only_adl, pp)), &ns::only_adl));
}

// Dependent launch arguments: the builtin defers until instantiation and must
// then resolve, including through ADL. While dependent, the builtin call is
// the auto* template argument of kernel_function, which must not assert
// (llvm/llvm-project#210524); the generic lambda covers a dependent context
// other than a function template.
template <typename T> auto launch_iota(T *p) {
  return nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(iota, T(1), p));
}
template <typename T> auto launch_only_adl(T *p) {
  return nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(only_adl, p));
}

void dependent(float *fp, ns::Params *pp) {
  static_assert(LAUNCHES(launch_iota(fp), &iota<float>));
  static_assert(LAUNCHES(launch_only_adl(pp), &ns::only_adl));
  auto generic = [](auto v, auto *p) {
    return nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(iota, v, p));
  };
  static_assert(LAUNCHES(generic(1.0f, fp), &iota<float>));
}

// Ill-formed launches. The best match for a long* argument is the overload
// that is not a kernel: the free function kernel check applies to the selected
// overload, not to the overload set.
ND_KERNEL void mixed(int *p) {}
void mixed(long *p) {} // expected-note {{declared here}}

ND_KERNEL void del(int *p) {} // expected-note {{candidate function not viable}}
void del(float *p) = delete;  // expected-note {{candidate function has been explicitly deleted}}

void negative(float *fp, long *lp) {
  // A failed or deleted call is diagnosed once, by overload resolution, with
  // no further diagnostic from the builtin.
  // expected-error@+2 {{no matching function for call to 'fill'}}
  // expected-note@#fill {{couldn't infer template argument 'Dim'}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill, fp, 1.0f));
  // expected-error@+1 {{the function resolved by '__builtin_sycl_launch_kernel' is not a SYCL free function kernel}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(mixed, lp));
  // expected-error@+1 {{call to deleted function 'del'}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(del, fp));
}

// A failure in a dependent context is diagnosed at instantiation.

template <typename T> void dependent_error(T *p) {
  // expected-error@+2 {{no matching function for call to 'fill'}}
  // expected-note@#fill {{couldn't infer template argument 'Dim'}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill, p, *p));
}
template void dependent_error(float *); // expected-note {{in instantiation of function template specialization 'dependent_error<float>' requested here}}
