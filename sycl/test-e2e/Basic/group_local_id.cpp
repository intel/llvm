// REQUIRES: cpu

// RUN: %{build} -std=c++17 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

using namespace sycl;

int main() {
  queue q(cpu_selector_v);

  const int sz = 16;
  q.submit([&](handler &h) {
    h.parallel_for(nd_range<1>{sz, sz}, [=](nd_item<1> item) {
      assert(item.get_local_id() == item.get_group().get_local_id());
    });
  });
  q.submit([&](handler &h) {
    h.parallel_for(
        nd_range<2>{range<2>{sz, sz}, range<2>{sz, sz}}, [=](nd_item<2> item) {
          assert(item.get_local_id() == item.get_group().get_local_id());
        });
  });
  q.submit([&](handler &h) {
    h.parallel_for(nd_range<3>{range<3>{sz, sz, sz}, range<3>{sz, sz, sz}},
                   [=](nd_item<3> item) {
                     assert(item.get_local_id() ==
                            item.get_group().get_local_id());
                   });
  });
  q.wait();
}
