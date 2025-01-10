// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} %t1 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_aot_flags -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t2
// RUN: %{run} %t2 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  constexpr std::size_t N = 8;
  constexpr std::size_t group_size = 4;

  sycl::queue Q;

  auto *array = sycl::malloc_device<int>(N, Q);

  std::vector<int> v(N);
  sycl::buffer<int, 1> buf(v.data(), v.size());

  Q.submit([&](sycl::handler &h) {
    auto buf_acc = buf.get_access<sycl::access::mode::read_write>(h);
    auto loc_acc = sycl::local_accessor<int>(group_size, h);
    h.parallel_for<class NoSanitized>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          auto gid = item.get_global_id(0);
          auto lid = item.get_local_id(0);
          array[gid] = buf_acc[gid] + loc_acc[lid];
        });
  });
  Q.wait();
  // CHECK-NOT: ERROR: DeviceSanitizer: out-of-bounds-access

  Q.submit([&](sycl::handler &h) {
    auto buf_acc = buf.get_access<sycl::access::mode::read_write>(h);
    auto loc_acc = sycl::local_accessor<int>(group_size, h);
    h.parallel_for<class Sanitized>(sycl::nd_range<1>(N, group_size),
                                    [=](sycl::nd_item<1> item) {
                                      auto gid = item.get_global_id(0);
                                      auto lid = item.get_local_id(0);
                                      array[gid] = buf_acc[gid] + loc_acc[lid];
                                    });
  });
  Q.wait();

  sycl::free(array, Q);
  std::cout << "PASS" << std::endl;
  // CHECK: PASS

  return 0;
}
