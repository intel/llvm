// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <cassert>

using namespace sycl;

// Check that information queries for dummy non-host events (e.g. USM operations
// for 0 bytes) work correctly.
int main() {
  queue q{{property::queue::enable_profiling()}};
  event e = q.memcpy(nullptr, nullptr, 0);

  assert(e.get_info<info::event::command_execution_status>() ==
         info::event_command_status::complete);
  assert(e.get_info<info::event::reference_count>() == 0);
  assert(e.get_profiling_info<sycl::info::event_profiling::command_submit>() ==
         0);
  assert(e.get_profiling_info<sycl::info::event_profiling::command_start>() ==
         0);
  assert(e.get_profiling_info<sycl::info::event_profiling::command_end>() == 0);
}
