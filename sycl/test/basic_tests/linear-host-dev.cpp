// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out | FileCheck %s

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Check that linear id is monotonically increased on host device.
// Only there we can reliable check that. Since the kernel has a restriction
// regarding usage of global variables, use stream to log the linear id
// and ensure that they're monotonically increased.
//
// Note: This test heavily relies on the current implementation of
// host device(single-threaded ordered execution). So if the implementation
// is somehow changed so it's no longer possible to run this test reliable
// it can be removed.

namespace s = cl::sycl;

int main(int argc, char *argv[]) {
  s::queue q;

  const size_t outer = 3;
  const size_t inner = 2;
  const s::range<2> rng = {outer, inner};

  q.submit([&](s::handler &h) {
    s::stream out(1024, 80, h);

    h.parallel_for<class linear_id>(s::range<2>(rng), [=](s::item<2> item) {
      // CHECK: 0
      // CHECK-NEXT: 1
      // CHECK-NEXT: 2
      // CHECK-NEXT: 3
      // CHECK-NEXT: 4
      // CHECK-NEXT: 5
      out << item.get_linear_id() << cl::sycl::endl;
    });
  });
  q.wait();

  return 0;
}
