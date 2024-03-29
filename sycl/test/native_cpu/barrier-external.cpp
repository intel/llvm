// REQUIRES: native_cpu_ock
// RUN: %clangxx -DFILE1 -fsycl -fsycl-targets=native_cpu %s -g -c -o %t1.o
// RUN: %clangxx -DFILE2 -fsycl -fsycl-targets=native_cpu %s -g -c -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t1.o %t2.o -g -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t
#include <sycl/sycl.hpp>

using namespace sycl;

#ifdef FILE1
SYCL_EXTERNAL void call_barrier(nd_item<1> &it) {
  it.barrier(access::fence_space::local_space);
}
#endif

#ifdef FILE2
SYCL_EXTERNAL void call_barrier(nd_item<1> &it);
class Test;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;
int main() {
  queue q;
  constexpr unsigned N = 10;
  constexpr unsigned NumG = 2;
  range<1> localR{N};
  range<1> globalR{NumG * N};
  buffer<int, 1> Buffer(globalR);
  q.submit([&](handler &h) {
    auto acc = Buffer.get_access<sycl_write>(h);
    h.parallel_for<Test>(nd_range<1>{globalR, localR}, [=](nd_item<1> it) {
      acc[it.get_global_id(0)] = 0;
      call_barrier(it);
      acc[it.get_global_id(0)]++;
    });
  });
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};
  for (unsigned i = 0; i < N * NumG; i++) {
    if (HostAccessor[i] != 1) {
      std::cout << "Error\n";
      return 1;
    }
  }
  std::cout << "Test passed\n";
  return 0;
}
#endif
