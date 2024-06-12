// REQUIRES: cuda
// RUN: %{build} %debug_option -o %t.out
// RUN: %{run} %t.out

// NOTE: Tests that the implicit global offset pass copies debug information

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue q;
  buffer<uint64_t, 1> t1(10);
  q.submit([&](handler &cgh) {
    auto table = t1.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class kernel>(10,
                                   [=](id<1> gtid) { table[gtid] = gtid[0]; });
  });
  q.wait();
}
