// REQUIRES: usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <CL/sycl.hpp>

constexpr int N = 5;

int main() {
  sycl::queue q;
  int *ptr = sycl::malloc_shared<int>(N, q);
  q.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> it) {
     auto g = it.get_group();
     auto mem = sycl::ext::oneapi::group_local_memory<int[N]>(g, 1, 2, 3, 4, 5);
     auto ref = *mem;
     for (int i = 0; i < N; ++i) {
       ptr[i] = ref[i];
     }
   }).wait();
  for (int i = 0; i < N; ++i) {
    assert(ptr[i] == (i + 1));
  }
  sycl::free(ptr, q);
}
