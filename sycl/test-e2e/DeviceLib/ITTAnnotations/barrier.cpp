// RUN: %{build} -fsycl-instrument-device-code -o %t.out
// RUN: %{run} %t.out

#include "sycl/detail/core.hpp"
#include <vector>

using namespace sycl;

int main() {
  queue q{};

  std::vector<int> data_vec(/*size*/ 10, /*value*/ 0);
  {
    range<1> num_items(data_vec.size());
    buffer<int> buf(data_vec.data(), num_items);
    range<1> local_range(2);

    // Ensure that a simple kernel gets run when instrumented with
    // ITT start/finish annotations and ITT wg_barrier/wi_resume annotations.
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      local_accessor<int, 1> local_acc(local_range, cgh);
      cgh.parallel_for<class simple_barrier_kernel>(
          nd_range<1>(num_items, local_range), [=](nd_item<1> item) {
            size_t idx = item.get_global_linear_id();
            int pos = idx & 1;
            int opp = pos ^ 1;
            local_acc[pos] = acc[idx];
            item.barrier(access::fence_space::local_space);
            acc[idx] = local_acc[opp];
          });
    });
  }

  return 0;
}
