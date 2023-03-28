// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  size_t N = 65536;
  size_t M = 1;

  q.parallel_for(sycl::range<2>{N, M}, [=](sycl::id<2> idx) {});
}
