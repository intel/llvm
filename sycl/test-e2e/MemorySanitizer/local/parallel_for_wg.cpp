// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -g -O0 -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O1 -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O2 -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -mllvm -msan-spir-locals=0 -g -O2 -o %t4.out
// RUN: %{run} %t4.out 2>&1 | FileCheck --check-prefixes DISABLE-CHECK %s
#include <sycl/group.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr std::size_t global_size = 4;
constexpr std::size_t local_size = 1;

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  auto data = sycl::malloc_device<int>(global_size, Q);

  Q.submit([&](handler &cgh) {
    cgh.parallel_for_work_group(
        range<1>(global_size), range<1>(local_size), [=](group<1> myGroup) {
          size_t j;
          myGroup.parallel_for_work_item([&](h_item<1> it) { data[j]++; });
        });
  });
  Q.wait();
  // CHECK-NOT: [kernel]
  // CHECK: DeviceSanitizer: use-of-uninitialized-value
  // CHECK: #0 {{.*}} {{.*parallel_for_wg.cpp}}:[[@LINE-6]]
  sycl::free(data, Q);

  std::cout << "PASS" << std::endl;
  return 0;
}
// DISABLE-CHECK-NOT: DeviceSanitizer: use-of-uninitialized-value
