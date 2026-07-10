// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: Default event constructor compatibility

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/sycl.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::event event1;
  sycl::event event2 = syclex::make_event();

  // Both should have complete status
  auto status1 = event1.get_info<sycl::info::event::command_execution_status>();
  auto status2 = event2.get_info<sycl::info::event::command_execution_status>();

  assert(status1 == sycl::info::event_command_status::complete);
  assert(status2 == sycl::info::event_command_status::complete);

  return 0;
}
