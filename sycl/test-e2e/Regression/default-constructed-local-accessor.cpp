// -O0 is necessary; on higher levels of optimization, an error
// would not occur because of dead argument elimination of the local_accessor.
// RUN: %{build} -o %t.out -O0
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;

using acc_t = local_accessor<int, 1>;

struct foo {
  acc_t acc;
  void operator()(nd_item<1>) const {}
};

int main() {
  queue q;
  q.submit([&](handler &cgh) { cgh.parallel_for(nd_range<1>(1, 1), foo{}); });
}
