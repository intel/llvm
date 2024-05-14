// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t
// This test is needed since we need to make sure that there no
// "multiple definitions" linker errors when a function appears
// both in the host and in the device module.
#include <sycl/sycl.hpp>

void increase(int *data, sycl::id<1> id, int val) { data[id] = data[id] + val; }

void init(int *data, sycl::id<1> id, int val) { data[id] = val; }

int main() {
  sycl::queue q;
  const size_t size = 10;
  int *data = sycl::malloc_device<int>(size, q);
  q.parallel_for(size, [=](sycl::id<1> id) { init(data, id, 41); }).wait();
  q.parallel_for(size, [=](sycl::id<1> id) { increase(data, id, 1); }).wait();
  int res[size];
  q.memcpy(res, data, size * sizeof(int)).wait();
  for (auto &el : res)
    std::cout << el << " ";
  std::cout << std::endl;

  sycl::free(data, q);
}
