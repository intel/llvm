// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Checks that khr::properties lists can be captured by, constructed in, and
// queried inside a kernel.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

void test(sycl::queue &q, int *out) {
  properties captured{kp::in_order{true}, kp::enable_profiling{false}};
  q.single_task([=] {
    properties local{kp::initialize_to_identity{false}};
    int r = 0;
    if constexpr (decltype(local)::has_property<
                      kp::key::initialize_to_identity>())
      r += local.get_property<kp::key::initialize_to_identity>().value;
    r += captured.get_property<kp::key::in_order>().value;
    r += captured.get_property<kp::key::enable_profiling>().value;
    *out = r;
  });
}
