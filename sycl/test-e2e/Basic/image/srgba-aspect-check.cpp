// RUN: %clangxx -fsycl  -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  // Ensure that plugins that don't implement the support can still handle the
  // aspect query.
  std::ignore = q.get_device().has(sycl::aspect::ext_oneapi_srgb);
  return 0;
}
