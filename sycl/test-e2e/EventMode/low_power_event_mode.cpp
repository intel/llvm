// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that the low-power event mode compiles and executes. Note that the
// event mode is a hint and has no observable behavior, aside from potential
// performance.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/event_mode_property.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;

  oneapiext::properties Props{
      oneapiext::event_mode{oneapiext::event_mode_enum::low_power}};

  sycl::event E = oneapiext::submit_with_event(
      Q, Props, [&](sycl::handler &CGH) { oneapiext::barrier(CGH); });

  oneapiext::submit_with_event(Q, Props, [&](sycl::handler &CGH) {
    oneapiext::partial_barrier(CGH, {E});
  }).wait_and_throw();

  return 0;
}
