// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  // Ensure that plugins that don't implement the support can still handle the
  // aspect query.
  std::ignore = q.get_device().has(sycl::aspect::ext_oneapi_srgb);
  return 0;
}
