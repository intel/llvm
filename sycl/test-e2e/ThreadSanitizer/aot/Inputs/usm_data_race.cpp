#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<char>(1, Q);
  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(32, 8),
                                [=](sycl::nd_item<1>) { array[0]++; });
   }).wait();
  // CHECK: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 1 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}usm_data_race.cpp:[[@LINE-4]]

  sycl::free(array, Q);
  return 0;
}
