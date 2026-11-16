// RUN: %{build} -O3 -DSYCL_DISABLE_IMAGE_ASPECT_WARNING=1 -o %t.out
// RUN: %{run} %t.out

// PERF-DESCRIPTION: Time of one sycl::device::has() call for every aspect.

#include "perf_harness.hpp"

#include <sycl/detail/core.hpp>

int main(int argc, char **argv) {
  constexpr int Calls = 100;

#define __SYCL_ASPECT(A, ID)                                                   \
  perf::add_case("has(" #A ")", "ns", [](perf::State &S) {                     \
    static sycl::device D;                                                     \
    S.items(Calls);                                                            \
    S.measure([&] {                                                            \
      for (int I = 0; I < Calls; ++I)                                          \
        (void)D.has(sycl::aspect::A);                                          \
    });                                                                        \
  });
#define __SYCL_ASPECT_DEPRECATED(A, ID, MSG) __SYCL_ASPECT(A, ID)
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT

  return perf::main(argc, argv);
}
