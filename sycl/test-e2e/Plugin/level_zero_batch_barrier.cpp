// REQUIRES: gpu, level_zero, level_zero_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 ZE_DEBUG=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

// Test that the wait with a barrier is fully batched, i.e. it doesn't cause
// extra submissions.

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace std;
using namespace sycl;

void submit_kernel(queue &q) {
  q.submit([&](auto &h) { h.parallel_for(1, [=](size_t id) {}); });
}

int main(int argc, char *argv[]) {
  queue q;

  submit_kernel(q); // starts a batch
                    // CHECK: ---> piEnqueueKernelLaunch
                    // CHECK-NOT: ZE ---> zeCommandQueueExecuteCommandLists

  // continue the batch
  event barrier = q.ext_oneapi_submit_barrier();
  // CHECK: ---> piEnqueueEventsWaitWithBarrier
  // CHECK-NOT: ZE ---> zeCommandQueueExecuteCommandLists

  submit_kernel(q);
  // CHECK: ---> piEnqueueKernelLaunch
  // CHECK-NOT: ZE ---> zeCommandQueueExecuteCommandLists

  // interop should close the batch
  ze_event_handle_t ze_event =
      get_native<backend::ext_oneapi_level_zero>(barrier);
  // CHECK: ---> piextEventGetNativeHandle
  // CHECK: ZE ---> zeCommandQueueExecuteCommandLists
  zeEventHostSynchronize(ze_event, UINT64_MAX);

  // CHECK: ---> piQueueFinish
  q.wait_and_throw();
  return 0;
}
