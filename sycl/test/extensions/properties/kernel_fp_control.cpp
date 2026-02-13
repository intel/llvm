// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

int main(void) {
  queue q;
  syclex::properties properties1{
      intelex::fp_control<intelex::fp_mode::round_downward |
                          intelex::fp_mode::denorm_ftz>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel1>(properties1, [=]() SYCL_ESIMD_KERNEL {});
  });

  syclex::properties properties2{
      intelex::fp_control<intelex::fp_mode::round_downward>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel2>(properties2, [=]() SYCL_ESIMD_KERNEL {});
  });

  syclex::properties properties3{
      intelex::fp_control<intelex::fp_mode::denorm_d_allow>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel3>(properties3, [=]() SYCL_ESIMD_KERNEL {});
  });

  // expected-error-re@sycl/ext/intel/experimental/fp_control_kernel_properties.hpp:* {{static assertion failed due to requirement {{.+}}: Mutually exclusive fp modes are specified for the kernel.}}
  syclex::properties properties4{
      intelex::fp_control<intelex::fp_mode::round_downward |
                          intelex::fp_mode::round_upward>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel4>(properties4, [=]() SYCL_ESIMD_KERNEL {});
  });

  // expected-error-re@sycl/ext/intel/experimental/fp_control_kernel_properties.hpp:* {{static assertion failed due to requirement {{.+}}: Mutually exclusive fp modes are specified for the kernel.}}
  syclex::properties properties5{
      intelex::fp_control<intelex::fp_mode::denorm_ftz |
                          intelex::fp_mode::denorm_hf_allow>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel5>(properties5, [=]() SYCL_ESIMD_KERNEL {});
  });

  // expected-error-re@sycl/ext/intel/experimental/fp_control_kernel_properties.hpp:* {{static assertion failed due to requirement {{.+}}: Mutually exclusive fp modes are specified for the kernel.}}
  syclex::properties properties6{
      intelex::fp_control<intelex::fp_mode::round_toward_zero |
                          intelex::fp_mode::denorm_ftz |
                          intelex::fp_mode::denorm_d_allow>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel6>(properties6, [=]() SYCL_ESIMD_KERNEL {});
  });

  return 0;
}
