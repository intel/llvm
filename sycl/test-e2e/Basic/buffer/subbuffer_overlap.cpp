// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<int, 1> b{1024};
  sycl::id<1> start_offset{64};
  size_t size = 16;
  sycl::buffer<int, 1> sub1{b, start_offset, sycl::range<1>{size}};
  sycl::buffer<int, 1> sub2{b, start_offset, sycl::range<1>{size * 2}};

  int idx = 0;
  for (auto &e : sycl::host_accessor{b})
    e = idx++ % size;

  // CHECK: piMemBufferPartition
  // CHECK: pi_buffer_region origin/size : 256/64
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{sub1, cgh};
    cgh.parallel_for(size, [=](auto id) { acc[id] += 1; });
  });
  // CHECK: piMemBufferPartition
  // CHECK: pi_buffer_region origin/size : 256/128
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{sub2, cgh};
    cgh.parallel_for(size * 2, [=](auto id) { acc[id] -= 1; });
  });

  // Print before asserts to ensure stream is flushed.
  for (auto &e : sycl::host_accessor{sub2})
    std::cout << e << " ";
  std::cout << std::endl;

  idx = 0;
  for (auto &e : sycl::host_accessor{sub2}) {
    assert(e == idx % size - idx / size);
    ++idx;
  }

  return 0;
}
