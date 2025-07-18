// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

struct ESIMDKernel {
  ESIMDKernel() {}

  void operator()(id<1> i) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        intelex::fp_control<intelex::fp_mode::round_downward |
                            intelex::fp_mode::denorm_ftz>};
  }
};

int main(void) {
  queue q;
  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.+}}: Floating point control property is supported for ESIMD kernels only.}}
  syclex::properties properties7{
      intelex::fp_control<intelex::fp_mode::round_toward_zero |
                          intelex::fp_mode::denorm_ftz>};
  q.submit([&](handler &cgh) {
    cgh.single_task<class TestKernel7>(properties7, [=]() {});
  });

  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.+}}: Floating point control property is supported for ESIMD kernels only.}}
  ESIMDKernel Kern;
  q.submit([&](handler &cgh) { cgh.parallel_for(range<1>(1), Kern); });

  return 0;
}
