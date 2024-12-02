// This test checks that with -fsycl-device-code-split=off, kernels
// with different reqd_work_group_size dimensions can be launched.

// RUN: %{build} -fsycl -fsycl-device-code-split=off -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip

#include <sycl/detail/core.hpp>

using namespace sycl;

int main(int argc, char **argv) {
  queue q;
  q.single_task([] {});
  q.parallel_for(range<2>(24, 1),
                 [=](auto) [[sycl::reqd_work_group_size(24, 1)]] {});
  return 0;
}
