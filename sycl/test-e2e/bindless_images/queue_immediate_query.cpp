// REQUIRES: level_zero

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/info/queue.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <iostream>

int main() {
  sycl::queue Q{sycl::ext::intel::property::queue::no_immediate_command_list()};
  assert(!Q.get_info<sycl::info::queue::ext_oneapi_immediate_command_list>());

  sycl::queue Q2{sycl::ext::intel::property::queue::immediate_command_list()};
  assert(Q2.get_info<sycl::info::queue::ext_oneapi_immediate_command_list>());
}
