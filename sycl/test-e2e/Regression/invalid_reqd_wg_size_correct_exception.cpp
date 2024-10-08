// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip

#include <sycl/detail/core.hpp>

int main() {
  try {
    sycl::queue q;
    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for(sycl::nd_range<1>({INT_MAX}, {INT_MAX}),
                        [=](auto item)
                            [[sycl::reqd_work_group_size(INT_MAX)]] {});
     }).wait_and_throw();
  } catch (sycl::exception &e) {
    assert(sycl::errc::kernel_not_supported == e.code());
  }
  return 0;
}
