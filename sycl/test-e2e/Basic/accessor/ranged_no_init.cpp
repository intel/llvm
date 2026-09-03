// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_HOST_UNIFIED_MEMORY=0 %{run} %t.out

// Regression test for a scheduler bug where sycl::property::no_init on a
// ranged accessor caused elements OUTSIDE the accessor's range to be
// discarded (zeroed) on devices without host-unified memory.
//
// SYCL 2020 §4.7.6.4 (Table 24, property::no_init):
//   "If this is a ranged accessor, this applies only to the elements within
//    the accessor's range. The values of unwritten elements outside of this
//    range are preserved."

#include <sycl/detail/core.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>

int main() {
  constexpr std::size_t n = 16, lo = 4, hi = 8;

  int observed[n];
  {
    sycl::buffer<int, 1> buf{sycl::range<1>(n)};

    {
      sycl::host_accessor h{buf, sycl::write_only};
      for (std::size_t i = 0; i < n; ++i)
        h[i] = 42;
    }

    sycl::queue{}
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<int, 1, sycl::access_mode::write> a{
              buf, cgh, sycl::range<1>(hi - lo), sycl::id<1>(lo),
              sycl::property_list{sycl::no_init}};
          cgh.parallel_for(sycl::range<1>(hi - lo),
                           [=](sycl::id<1> i) { a[i] = 7; });
        })
        .wait_and_throw();

    sycl::host_accessor h{buf, sycl::read_only};
    for (std::size_t i = 0; i < n; ++i)
      observed[i] = h[i];
  }

  int failures = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int expected = (i >= lo && i < hi) ? 7 : 42;
    if (observed[i] != expected) {
      std::cerr << "  index " << i << ": got " << observed[i] << ", expected "
                << expected << "\n";
      ++failures;
    }
  }
  if (failures) {
    std::cerr << failures
              << " element(s) outside the ranged no_init accessor were not "
                 "preserved.\n";
    return 1;
  }
  return 0;
}
