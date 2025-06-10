#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

__attribute__((noinline)) void foo(char *array, int val) { *array += val; }

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<char>(1, Q);
  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(128, 8),
                                [=](sycl::nd_item<1> it) {
                                  *array += it.get_global_linear_id();
                                  foo(array, it.get_local_linear_id());
                                });
   }).wait();
  // CHECK: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 1 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}usm_data_race.cpp

  sycl::free(array, Q);
  return 0;
}
