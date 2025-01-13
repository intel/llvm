// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  auto device = sycl::device();
#ifndef SYCL_EXT_ONEAPI_QUEUE_PRIORITY
  return 0;
#else
  auto queue =
      sycl::queue(device, {sycl::ext::oneapi::property::queue::priority_low()});
  auto queue2 = sycl::queue(
      device, {sycl::ext::oneapi::property::queue::priority_normal()});
  auto queue3 = sycl::queue(
      device, {sycl::ext::oneapi::property::queue::priority_high()});

#endif
  return 0;
}
