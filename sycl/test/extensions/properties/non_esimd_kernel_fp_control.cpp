// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=%sycl_triple -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

struct ESIMDKernel {
  ESIMDKernel() {}

  void operator()(id<1> i) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        intelex::fp_control<intelex::fp_mode::round_downward |
                            intelex::fp_mode::denorm_ftz>};
  }
};

int main(void) {
  queue q;
  // expected-error@* {{SYCL fp_control control property is supported only for ESIMD kernels}}
  syclex::properties properties7{
      intelex::fp_control<intelex::fp_mode::round_toward_zero |
                          intelex::fp_mode::denorm_ftz>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel7>(properties7, [=]() {});
  });

  // expected-error@* {{SYCL fp_control control property is supported only for ESIMD kernels}}
  ESIMDKernel Kern;
  q.submit([&](handler &cgh) { cgh.parallel_for(range<1>(1), Kern); });

  return 0;
}