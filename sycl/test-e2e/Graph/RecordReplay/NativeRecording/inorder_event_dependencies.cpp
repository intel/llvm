// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_UR_TRACE=2 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// The SYCL RT prunes redundant in-order wait events on the same queue, but must
// disable this while any native recording is active for proper handling in the
// native runtime. Removing this is relevant for external graph events. SYCL is
// unaware of queue recording status without querying the native runtime (e.g.
// L0), so we instead conservatively disable this optimization when any native
// recording is active on the queue's context. This avoids querying through the
// native runtime on the hot kernel submission path.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, property::queue::in_order{}};
  queue Queue2{Ctx, Dev, property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *Data1 = malloc_device<int>(N, Queue1);
  int *Data2 = malloc_device<int>(N, Queue1);
  int *Data3 = malloc_device<int>(N, Queue1);
  int *Data4 = malloc_device<int>(N, Queue1);
  int *Data5 = malloc_device<int>(N, Queue1);

  QueueStateVerifier verifier(Queue1, Queue2);
  verifier.verify(EXECUTING, EXECUTING);

  // --- Case 1: optimization ON - SYCL RT removes event ---
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  event Event1 =
      Queue1.parallel_for(range<1>{N}, [=](id<1> idx) { Data1[idx] = 5; });
  // Same-queue in-order dependency is redundant, so it is pruned.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  Queue1.submit([&](handler &CGH) {
    CGH.depends_on(Event1);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data1[idx] += 1; });
  });

  // --- Case 2 (cross-boundary): pre-recording event used during recording ---
  // Capture a no-op event on Queue1 before recording begins. It is used as a
  // wait event during recording below.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  event PreRecordEvent = Queue1.parallel_for(range<1>{N}, [=](id<1>) {});

  // CHECK: <--- urQueueBeginCaptureIntoGraphExp
  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  // Cross-boundary: wait during recording on the pre-recording event from
  // Queue1. This event is not marked as an external wait, and it is expected
  // to become an error in the native runtime.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 1
  try {
    Queue1.submit([&](handler &CGH) {
      CGH.depends_on(PreRecordEvent);
      CGH.parallel_for(range<1>{N}, [=](id<1>) {});
    });
  } catch (const exception &) {
  }

  // --- Case 3: optimization OFF via the context recording counter ---
  // Recording Queue1 bumps its context's recording counter. Queue2 shares that
  // same context, so the optimization is disabled for it too, even though
  // Queue2 is not yet recording or forked.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  event Event2 =
      Queue2.parallel_for(range<1>{N}, [=](id<1> idx) { Data2[idx] = 5; });
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 1
  Queue2.submit([&](handler &CGH) {
    CGH.depends_on(Event2);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data2[idx] += 1; });
  });

  // Implicit in-order deps should still show 0 wait events.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  Queue1.parallel_for(range<1>{N}, [=](id<1> idx) { Data5[idx] = 5; });
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  Queue1.parallel_for(range<1>{N}, [=](id<1> idx) { Data5[idx] += 1; });

  // --- Case 4: fork to Queue2, optimization still OFF ---
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  event Fork =
      Queue1.parallel_for(range<1>{N}, [=](id<1> idx) { Data3[idx] = 5; });

  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 1
  event ForkConsumer = Queue2.submit([&](handler &CGH) {
    CGH.depends_on(Fork);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data3[idx] += 10; });
  });
  verifier.verify(RECORDING, RECORDING);

  // Same-queue dependency on the forked Queue2, optimization still off.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 1
  event Join = Queue2.submit([&](handler &CGH) {
    CGH.depends_on(ForkConsumer);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data3[idx] += 1; });
  });

  // Join Queue2 back into Queue1 to close the native graph cleanly.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 1
  Queue1.submit([&](handler &CGH) {
    CGH.depends_on(Join);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data3[idx] += 100; });
  });

  // --- Case 5: optimization ON again after recording ended ---
  // CHECK: <--- urQueueEndGraphCaptureExp
  Graph.end_recording(Queue1);
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();
  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  event Event4 =
      Queue1.parallel_for(range<1>{N}, [=](id<1> idx) { Data4[idx] = 5; });
  // Same-queue dependency created with no recording active: pruned again.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 0
  Queue1.submit([&](handler &CGH) {
    CGH.depends_on(Event4);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data4[idx] += 1; });
  });

  // --- Case 6 (cross-boundary): recorded event used after recording ---
  // Wait on Queue2's Join event from the recording region. The dependency is
  // retained. This is not an external graph event, and it is expected to become
  // an error in the native runtime.
  // CHECK: <--- urEnqueueKernelLaunchWithArgsExp
  // CHECK-SAME: .numEventsInWaitList = 1
  try {
    Queue2.submit([&](handler &CGH) {
      CGH.depends_on(Join);
      CGH.parallel_for(range<1>{N}, [=](id<1>) {});
    });
  } catch (const exception &) {
  }

  Queue1.wait();
  Queue2.wait();

  std::vector<int> Host1(N), Host2(N), Host3(N), Host4(N), Host5(N);
  Queue1.memcpy(Host1.data(), Data1, N * sizeof(int));
  Queue1.memcpy(Host2.data(), Data2, N * sizeof(int));
  Queue1.memcpy(Host3.data(), Data3, N * sizeof(int));
  Queue1.memcpy(Host4.data(), Data4, N * sizeof(int));
  Queue1.memcpy(Host5.data(), Data5, N * sizeof(int));
  Queue1.wait();

  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, 6, Host1[i], "Data1"));
    assert(check_value(i, 6, Host2[i], "Data2"));
    assert(check_value(i, 116, Host3[i], "Data3"));
    assert(check_value(i, 6, Host4[i], "Data4"));
    assert(check_value(i, 6, Host5[i], "Data5"));
  }

  free(Data1, Queue1);
  free(Data2, Queue1);
  free(Data3, Queue1);
  free(Data4, Queue1);
  free(Data5, Queue1);

  return 0;
}
