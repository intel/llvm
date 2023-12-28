// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %s
#include <cstddef>
#include <sycl/sycl.hpp>

constexpr std::size_t N = 1024ULL;
constexpr std::size_t group_size = 4;

int k_func(sycl::nd_item<2> item) {
  auto mem1 = sycl::ext::oneapi::group_local_memory<int[5]>(item.get_group(), 1,
                                                            2, 3, 4, 5);
  auto mem2 = sycl::ext::oneapi::group_local_memory<int[5]>(item.get_group(), 1,
                                                            2, 3, 4, 5);
  auto mem3 = sycl::ext::oneapi::group_local_memory<int[5]>(item.get_group(), 1,
                                                            2, 3, 4, 5);
  auto ref1 = *mem1, ref2 = *mem2, ref3 = *mem3;

  int sum = 0;
  sum += ref1[0];
  sum += ref2[3];
  for (int i = 0; i < 10; ++i) {
    sum += ref3[i];
    // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
    // CHECK: {{READ of size 1 at kernel <.*MyKernel> LID\(0, 1, 0\) GID\(.*, .*, 0\)}}
    // CHECK: {{  #0 .* .*local-overflow-4.cpp:}}[[@LINE-3]]
  }

  return sum;
}

int main() {
  sycl::queue Q;
  int *ptr = sycl::malloc_shared<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class MyKernel>(
         sycl::nd_range<2>({N, N}, {group_size, group_size}),
         [=](sycl::nd_item<2> item) {
           auto wgid = item.get_group_linear_id();
           // *ptr = wgid;
           // sycl::ext::oneapi::experimental::printf("wgid: %u\n",
           //                                         wgid);
           auto i = item.get_local_id(0);
           if (i % group_size == 1) {
             *ptr += k_func(item);
           }
         });
   }).wait();

  return 0;
}
