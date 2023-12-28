// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %s
#include <sycl.hpp>

constexpr std::size_t N = 16;
constexpr std::size_t group_size = 4;

int main() {
  sycl::queue q;
  int *ptr = sycl::malloc_shared<int>(N, q);

  q.parallel_for<class MyKernel>(
       sycl::nd_range<1>{N, group_size},
       [=](sycl::nd_item<1> it) {
         auto g = it.get_group();
         auto mem1 =
             sycl::ext::oneapi::group_local_memory<int[N]>(g, 1, 2, 3, 4, 5);
         auto mem2 =
             sycl::ext::oneapi::group_local_memory<int[N]>(g, 1, 2, 3, 4, 5);
         auto ref1 = *mem1, ref2 = *mem2;
         for (int i = 0; i < N + 1; ++i) {
           ptr[i] = ref1[i];
           // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
           // CHECK: {{READ of size 1 at kernel <.*MyKernel> LID\(0, 0, 0\) GID\(.*, 0, 0\)}}
           // CHECK: {{  #0 .* .*local-overflow-3.cpp:}}[[@LINE-3]]
         }
         for (int i = 0; i < N; ++i) {
           ptr[i] += ref2[i];
         }
       })
      .wait();

  sycl::free(ptr, q);

  return 0;
}
