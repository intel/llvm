// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-host -fsyntax-only -include-internal-header %S/Inputs/resolve_kernel_function_int_header.h %s -verify=host
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-host -fsyntax-only -include-internal-header %S/Inputs/resolve_kernel_function_int_header.h -fdelayed-template-parsing %s -verify=host
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsyntax-only %s -verify=device

// The integration header makes every free function kernel visible on the host
// side ahead of the application's code. A kernel that the call selects only
// through those forward declarations is rejected, because the device side
// cannot see it at the launch and would select a different kernel. With
// delayed template parsing, the default for Windows hosts, a template body is
// parsed at the end of the translation unit on the host side only.

#include "sycl_kernel_function_macro.hpp"

ND_KERNEL void before(int *p, double v) {}
ND_KERNEL void late(int *p, double v) {}
ND_KERNEL void dep(int *p, double v) {}

template <typename T> void launch_nondep(int *p) {
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'late' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(late, p, 1.0f));
}

template <typename T> void launch_dep(int *p, T v) {
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'dep' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(dep, p, v));
}

// Explicit specializations and instantiations are resolved consistently: the
// integration header forward declares the explicit specialization, which is
// the kernel both sides select.
template <typename T> ND_KERNEL void ffk(int *p, T v) {}
template <> ND_KERNEL void ffk(int *p, int v) {}
template void ffk(int *p, long);

// An explicit specialization declared after its use is ill-formed C++; the
// forward declaration in the integration header hides that on the host side,
// but the device side diagnoses it.
template <typename T> ND_KERNEL void spec_after_use(int *p, T v) {}

// A better-matching non-template overload declared after the launch.
template <typename T> ND_KERNEL void overload_after_use(int *p, T v) {}

// A better-matching template declared after the launch, which another launch
// uses, so that the integration header forward declares it.
template <typename T> ND_KERNEL void template_after_use(int *p, T v) {}

void test_templates(int *p, float *fp, double *dp) {
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ffk, p, '1')), &ffk<char>));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ffk, p, 2)), &ffk<int>));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ffk, p, 3L)), &ffk<long>));

  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(spec_after_use, p, 2)); // #late_spec_use

  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'overload_after_use' of type 'void (int *, double *)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(overload_after_use, p, dp));

  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'template_after_use<float>' of type 'void (int *, float *)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(template_after_use, p, fp));
}

// device-error@+2 {{explicit specialization of 'spec_after_use<int>' after instantiation}}
// device-note@#late_spec_use {{implicit instantiation first required here}}
template <> ND_KERNEL void spec_after_use(int *p, int v) {}
// host-note@+1 {{'overload_after_use' should be declared prior to the call site}}
ND_KERNEL void overload_after_use(int *p, double *v) {}
// host-note@+1 {{'template_after_use<float>' should be declared prior to the call site}}
template <typename T> ND_KERNEL void template_after_use(int *p, T *v) {}

void test_templates_after(int *p, float *fp) {
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(template_after_use, p, fp)), static_cast<void (*)(int *, float *)>(&template_after_use<float>)));
}

// A declaration that precedes the launch but is not visible there: a
// using-declaration only names the overloads declared before it, and a
// block-scope declaration is visible only in its block.
namespace ns {
ND_KERNEL void via_using(int *p, double v) {}
} // namespace ns
using ns::via_using; // host-note {{using declaration}}
namespace ns {
ND_KERNEL void via_using(int *p, float v) {}
} // namespace ns

ND_KERNEL void block_scope(int *p, double v) {}
void other() { ND_KERNEL void block_scope(int *p, float v); }

void test_visibility(int *p) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(via_using, p, 1.0f)), static_cast<void (*)(int *, double)>(&ns::via_using)));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(block_scope, p, 1.0f)), static_cast<void (*)(int *, double)>(&block_scope)));
#endif
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'via_using' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(via_using, p, 1.0f));
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'block_scope' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(block_scope, p, 1.0f));
}

// host-note@+1 {{'block_scope' should be declared prior to the call site}}
ND_KERNEL void block_scope(int *p, float v) {}

void test(int *p) {
  // Declared before the launch: both sides select the same kernel.
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(before, p, 1.0f)), &before));

  // The device side selects late(int *, double).
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(late, p, 1.0f)), static_cast<void (*)(int *, double)>(&late)));
#endif
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'late' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(late, p, 1.0f));
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'late' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  (void)__builtin_sycl_resolve_kernel_function(late(p, 1.0f));

  // On the device side the template is not declared at all yet.
  // device-error@+2 {{use of undeclared identifier 'late_tmpl'}}
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'late_tmpl<float>' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(late_tmpl, p, 1.0f));

  launch_dep(p, 1.0f);
  launch_nondep<int>(p);
}

// host-note@+1 3 {{'late' should be declared prior to the call site}}
ND_KERNEL void late(int *p, float v) {}
// host-note@+1 {{'late_tmpl<float>' should be declared prior to the call site}}
template <typename T> ND_KERNEL void late_tmpl(int *p, T v) {}
// host-note@+1 {{'dep' should be declared prior to the call site}}
ND_KERNEL void dep(int *p, float v) {}
