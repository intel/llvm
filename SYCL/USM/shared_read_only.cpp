// REQUIRES: level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER  %t.out 2> %t1.out; cat %t1.out %GPU_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  auto *p1 = sycl::malloc_shared<float>(
      42, q, {sycl::ext::oneapi::property::usm::device_read_only()});
  auto *p2 = sycl::malloc_shared<float>(42, q);

  // CHECK: zeMemAllocShared
  // CHECK: {{zeCommandListAppendMemAdvise.*ZE_MEMORY_ADVICE_SET_READ_MOSTLY}}
  // CHECK: {{zeCommandListAppendMemAdvise.*ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION*}}
  // CHECK: zeMemAllocShared
  // CHECK-NOT: MemAdvise

  free(p2, q);
  free(p1, q);
  return 0;
}
