// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-host -fsyntax-only -include-internal-header %S/Inputs/resolve_kernel_function_int_header.h %s -verify=host
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsyntax-only %s -verify=device

// The integration header makes every free function kernel visible on the host
// side ahead of the application's code. A kernel that the call selects only
// through those forward declarations is rejected, because the device side
// cannot see it at the launch and would select a different kernel.

#include "sycl_kernel_function_macro.hpp"

ND_KERNEL void before(int *p, double v) {}
ND_KERNEL void late(int *p, double v) {}
ND_KERNEL void dep(int *p, double v) {}

template <typename T> void launch_dep(int *p, T v) {
  // host-error@+1 {{SYCL-DPCPP split compilation selects free function kernel 'dep' of type 'void (int *, float)', which the device compiler cannot see at this launch}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(dep, p, v));
}

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
}

// host-note@+1 2 {{'late' should be declared prior to the call site}}
ND_KERNEL void late(int *p, float v) {}
// host-note@+1 {{'late_tmpl<float>' should be declared prior to the call site}}
template <typename T> ND_KERNEL void late_tmpl(int *p, T v) {}
// host-note@+1 {{'dep' should be declared prior to the call site}}
ND_KERNEL void dep(int *p, float v) {}
