// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <CL/sycl.hpp>

constexpr int N = 5;

int main() {
  sycl::queue q;
  sycl::buffer<int> buf{sycl::range{N}};
  q.submit([&](sycl::handler &h) {
    sycl::accessor acc{buf, h};
    h.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> it) {
      auto g = it.get_group();
      auto mem =
          sycl::ext::oneapi::group_local_memory<int[N]>(g, 1, 2, 3, 4, 5);
      auto ref = *mem;
      for (int i = 0; i < N; ++i) {
        acc[i] = ref[i];
      }
    });
  });
  sycl::host_accessor result{buf};
  for (int i = 0; i < N; ++i) {
    assert(result[i] == (i + 1));
  }
}
