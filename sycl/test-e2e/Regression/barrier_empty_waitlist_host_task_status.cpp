// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP=1 %{run} %t.out

// Regression test for CMPLRLLVM-74969.
//
// A barrier submitted through the handler API with an *empty* wait list but
// with an explicit handler::depends_on() on a blocked host task must report
// info::event_command_status::complete once the queue has been waited on.
// It used to stay stuck at 'submitted' because the scheduler path for an
// empty-wait-list barrier returned early from ExecCGCommand::enqueueImpQueue()
// without ever assigning a native event handle to the barrier event. With no
// handle, event_impl::get_info<command_execution_status>() falls back to
// reporting 'submitted' whenever MCommand is still set.
//
// Graph cleanup is what clears MCommand, so without it the bug is racy (it
// reproduced ~5% of the time). The second RUN line disables cleanup to pin the

// A barrier submitted through the handler API with an *empty* wait list but
// with an explicit handler::depends_on() on a blocked host task must report
// info::event_command_status::complete once the queue has been waited on.

#include <sycl/detail/core.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <vector>

int main() {
  sycl::queue Q;

  // 1. Submit a host task that blocks until we release the mutex.
  std::mutex Mtx;
  std::unique_lock<std::mutex> MainLock{Mtx};
  sycl::event HostTaskEvent = Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Wait{Mtx}; });
  });

  // 2. Submit a barrier with an empty wait list that explicitly depends on the
  //    blocked host task.
  sycl::event BarrierEvent = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostTaskEvent);
    CGH.ext_oneapi_barrier(std::vector<sycl::event>{});
  });

  // 3. Unblock the host task.
  MainLock.unlock();

  // 4. Wait for everything to drain.
  Q.wait();

  // 5. Both events must now report 'complete'.
  auto HostStatus =
      HostTaskEvent.get_info<sycl::info::event::command_execution_status>();
  auto BarrierStatus =
      BarrierEvent.get_info<sycl::info::event::command_execution_status>();

  auto ToStr = [](sycl::info::event_command_status S) {
    switch (S) {
    case sycl::info::event_command_status::submitted:
      return "submitted";
    case sycl::info::event_command_status::running:
      return "running";
    case sycl::info::event_command_status::complete:
      return "complete";
    default:
      return "ext_oneapi_unknown";
    }
  };

  int Failures = 0;
  if (HostStatus != sycl::info::event_command_status::complete) {
    std::cout << "FAIL: host task event status is " << ToStr(HostStatus)
              << ", expected complete\n";
    ++Failures;
  }
  if (BarrierStatus != sycl::info::event_command_status::complete) {
    std::cout << "FAIL: barrier event status is " << ToStr(BarrierStatus)
              << ", expected complete\n";
    ++Failures;
  }

  if (!Failures)
    std::cout << "PASS\n";
  return Failures != 0;
}
