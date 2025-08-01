// REQUIRES: gpu, level_zero, level_zero_dev_kit
// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-INTENDED: v2 adapter does not support regular cmd lists

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_UR_TRACE=2 UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.out 2>&1 | FileCheck %s

// Test that the wait with a barrier is fully batched, i.e. it doesn't cause
// extra submissions.

#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <vector>

using namespace std;
using namespace sycl;

void submit_kernel(queue &q) {
  q.submit([&](auto &h) { h.parallel_for(1, [=](size_t id) {}); });
}

int main(int argc, char *argv[]) {
  queue q;

  submit_kernel(q); // starts a batch
                    // CHECK: ---> urEnqueueKernelLaunchWithArgsExp
                    // CHECK-NOT: zeCommandQueueExecuteCommandLists

  // Initialize Level Zero driver is required if this test is linked
  // statically with Level Zero loader, the driver will not be init otherwise.
  ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed\n";
    return 1;
  }

  // continue the batch
  event barrier = q.ext_oneapi_submit_barrier();
  // CHECK: ---> urEnqueueEventsWaitWithBarrierExt
  // CHECK-NOT: zeCommandQueueExecuteCommandLists

  submit_kernel(q);
  // CHECK: ---> urEnqueueKernelLaunchWithArgsExp
  // CHECK-NOT: zeCommandQueueExecuteCommandLists

  // interop should close the batch
  ze_event_handle_t ze_event =
      get_native<backend::ext_oneapi_level_zero>(barrier);
  // CHECK: ---> urEventGetNativeHandle
  // CHECK: zeCommandQueueExecuteCommandLists
  zeEventHostSynchronize(ze_event, UINT64_MAX);

  // CHECK: ---> urQueueFinish
  q.wait_and_throw();
  return 0;
}
