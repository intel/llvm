// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-host -fsyntax-only -DINTEGRATION_HEADER %s -verify=host
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsyntax-only %s -verify=device

// On the host side the integration header forward declares each kernel without
// its properties ahead of the application's declaration, so whether the
// resolved function is a free function kernel is checked only on the device
// side. Both sides resolve the same kernel.

#include "sycl_kernel_function_macro.hpp"

// host-no-diagnostics

#ifdef INTEGRATION_HEADER
void k(int *p);
void k(float *p);
#endif

ND_KERNEL void k(int *p) {}
ND_KERNEL void k(float *p) {}

void not_a_kernel(int *p) {} // device-note {{declared here}}

void test(int *ip, float *fp) {
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(k, ip)), static_cast<void (*)(int *)>(&k)));
  static_assert(LAUNCHES(nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(k, fp)), static_cast<void (*)(float *)>(&k)));

  // device-error@+1 {{is not a SYCL free function kernel}}
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(not_a_kernel, ip));
}
