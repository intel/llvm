// RUN: %{build} -O3 -DSYCL_DISABLE_IMAGE_ASPECT_WARNING=1 -o %t.out
// RUN: %{run} %t.out
#include "../perf_harness.hpp"

#include <sycl/detail/core.hpp>

int main(int argc, char **argv) {
  perf::Benchmark Bench(
      argc, argv, "Aspect queries",
      "Time of one sycl::device::has() call for every aspect.", perf::Unit::ns);
  sycl::device d;

#define ASPECT(A)                                                              \
  Bench.add_perf_test("aspect::" #A, [&] { (void)d.has(sycl::aspect::A); });

#define __SYCL_ASPECT(A, ID) ASPECT(A)
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MSG) __SYCL_ASPECT(ASPECT, ID)
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
}
