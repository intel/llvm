// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// A barrier submitted through the handler API with an *empty* wait list but
// with an explicit handler::depends_on() must report
// info::event_command_status::complete once the queue has been waited on, and
// must still honor that dependency.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <mutex>
#include <vector>

static const char *toStr(sycl::info::event_command_status S) {
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
}

static int checkComplete(const char *Name, sycl::event &E) {
  auto S = E.get_info<sycl::info::event::command_execution_status>();
  if (S == sycl::info::event_command_status::complete)
    return 0;
  std::cout << "FAIL: " << Name << " status is " << toStr(S)
            << ", expected complete\n";
  return 1;
}

// Empty barrier wait list plus a host task dependency. The host task event has
// no native event, so this exercises the case where the barrier ends up with
// nothing at all to wait for.
static int testHostTaskDependency() {
  sycl::queue Q;

  std::mutex Mtx;
  std::unique_lock<std::mutex> MainLock{Mtx};
  sycl::event HostTaskEvent = Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Wait{Mtx}; });
  });

  sycl::event BarrierEvent = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostTaskEvent);
    CGH.ext_oneapi_barrier(std::vector<sycl::event>{});
  });

  MainLock.unlock();
  Q.wait();

  return checkComplete("host task event", HostTaskEvent) +
         checkComplete("barrier event (host task dep)", BarrierEvent);
}

// Empty barrier wait list plus a *native* dependency coming from another queue.
// Unlike a host task event, this one is forwarded to the barrier through the UR
// wait list, so it covers the dependency-forwarding path. The dependency is
// gated behind a blocked host task, which lets us check that the barrier cannot
// complete while the dependency is still outstanding.
static int testNativeCrossQueueDependency() {
  sycl::queue Q1;
  sycl::queue Q2{Q1.get_context(), Q1.get_device()};

  constexpr size_t N = 64;
  int *Ptr = sycl::malloc_device<int>(N, Q2);

  std::mutex Mtx;
  std::unique_lock<std::mutex> MainLock{Mtx};
  sycl::event Gate = Q2.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Wait{Mtx}; });
  });

  // A device command, so its event carries a native handle once enqueued.
  sycl::event DeviceEvent = Q2.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Gate);
    CGH.memset(Ptr, 0, N * sizeof(int));
  });

  sycl::event BarrierEvent = Q1.submit([&](sycl::handler &CGH) {
    CGH.depends_on(DeviceEvent);
    CGH.ext_oneapi_barrier(std::vector<sycl::event>{});
  });

  int Failures = 0;

  // The gate still holds the lock, so DeviceEvent cannot have run yet. A
  // barrier that dropped its depends_on() dependency could report complete
  // here.
  auto Early =
      BarrierEvent.get_info<sycl::info::event::command_execution_status>();
  if (Early == sycl::info::event_command_status::complete) {
    std::cout << "FAIL: barrier completed while its dependency was still "
                 "blocked\n";
    ++Failures;
  }

  MainLock.unlock();
  Q1.wait();
  Q2.wait();

  Failures += checkComplete("device event", DeviceEvent);
  Failures += checkComplete("barrier event (native dep)", BarrierEvent);

  sycl::free(Ptr, Q2);
  return Failures;
}

int main() {
  int Failures = testHostTaskDependency() + testNativeCrossQueueDependency();
  if (!Failures)
    std::cout << "PASS\n";
  return Failures != 0;
}
