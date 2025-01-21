// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>

using namespace sycl;

void test(queue q) {
  int out = 0;
  size_t G = 4;

  range<2> R(G, G);
  {
    buffer<int> out_buf(&out, 1);

    q.submit([&](handler &cgh) {
      auto out = out_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(nd_range<2>(R, R), [=](nd_item<2> it) {
        group<2> g = it.get_group();
        if (g.leader()) {
          out[0] += 1;
        }
      });
    });
  }
  assert(out == 1);
}

int main() {
  queue q;
  test(q);

  std::cout << "Test passed." << std::endl;
  return 0;
}
