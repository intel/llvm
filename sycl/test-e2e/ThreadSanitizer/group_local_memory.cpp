// REQUIRES: linux, cpu || (gpu && level_zero)
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O0 -g -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s
#include "sycl/ext/oneapi/group_local_memory.hpp"
#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

__attribute__((noinline)) void check(int *ptr, size_t val) { *ptr += val; }

int main() {
  sycl::queue Q;
  auto *sum = sycl::malloc_shared<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class Test>(
        sycl::nd_range<1>(128, 16), [=](sycl::nd_item<1> item) {
          auto ptr =
              sycl::ext::oneapi::group_local_memory<int>(item.get_group());
          *ptr += item.get_global_linear_id();

          check(ptr, item.get_local_linear_id());

          item.barrier();

          if (item.get_global_linear_id() == 0)
            *sum = *ptr;
        });
  });
  Q.wait();
  // CHECK: WARNING: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 4 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}group_local_memory.cpp

  sycl::free(sum, Q);
  return 0;
}
