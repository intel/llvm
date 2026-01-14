// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>

int main() {
  sycl::device dev;
  auto ctx1 = dev.ext_oneapi_get_default_context();
  auto ctx2 = dev.get_platform().khr_get_default_context();
  return !(ctx1 == ctx2);
}
