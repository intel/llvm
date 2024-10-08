// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  size_t N = 65536;
  size_t M = 1;

  q.parallel_for(sycl::range<2>{N, M}, [=](sycl::id<2> idx) {});
}
