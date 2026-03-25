// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: level_zero_dev_kit

// RUN: %{build} -lze_loader -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test native recording with Level Zero memory operations.
// Records L0 memset, device-to-device copy, and device-to-host copy
// directly to the recording command list over an in-order queue.

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"

#include <level_zero/ze_api.h>
#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  const size_t N = 1024;
  uint32_t *DeviceSrc = malloc_device<uint32_t>(N, Queue);
  uint32_t *DeviceDst = malloc_device<uint32_t>(N, Queue);
  uint32_t *HostDst = malloc_host<uint32_t>(N, Queue);

  for (size_t i = 0; i < N; i++) {
    HostDst[i] = 0;
  }

  ze_command_list_handle_t ZeCommandList;
  bool success = getCommandListFromQueue(Queue, ZeCommandList);
  assert(success);

  exp_ext::command_graph Graph{
      Context, Device, {exp_ext::property::graph::enable_native_recording{}}};

  CommandListStateVerifier verifier(ZeCommandList);
  verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);

  verifier.verify(RECORDING);

  // 1. Level Zero memset - fill DeviceSrc with pattern 0x42 (byte pattern)
  uint32_t Pattern = 0x42;
  ASSERT_ZE_RESULT_SUCCESS(zeCommandListAppendMemoryFill(
      ZeCommandList, DeviceSrc, &Pattern, sizeof(uint32_t),
      N * sizeof(uint32_t), nullptr, 0, nullptr));

  // 2. Level Zero device-to-device copy - copy DeviceSrc to DeviceDst
  ASSERT_ZE_RESULT_SUCCESS(
      zeCommandListAppendMemoryCopy(ZeCommandList, DeviceDst, DeviceSrc,
                                    N * sizeof(uint32_t), nullptr, 0, nullptr));

  // 3. Level Zero device-to-host copy - copy DeviceDst to HostDst
  ASSERT_ZE_RESULT_SUCCESS(
      zeCommandListAppendMemoryCopy(ZeCommandList, HostDst, DeviceDst,
                                    N * sizeof(uint32_t), nullptr, 0, nullptr));

  Graph.end_recording(Queue);

  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  // Verify results on host
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Pattern, HostDst[i], "HostDst"));
  }

  free(DeviceSrc, Queue);
  free(DeviceDst, Queue);
  free(HostDst, Queue);

  return 0;
}
