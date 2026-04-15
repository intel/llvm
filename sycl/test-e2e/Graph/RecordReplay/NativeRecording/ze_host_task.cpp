// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"

#include <level_zero/ze_api.h>
#include <sycl/properties/all_properties.hpp>

constexpr size_t N = 1024;

void ZE_APICALL HostFunction(void *UserData) {
  uint32_t *Data = static_cast<uint32_t *>(UserData);
  for (size_t i = 0; i < N; i++) {
    Data[i] = Data[i] * 3;
  }
}

int main() {
  queue Queue{property::queue::in_order{}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  ze_driver_handle_t ZeDriver;
  ASSERT_ZE_RESULT_SUCCESS(getDriver(ZeDriver));

  zeCommandListAppendHostFunction_fn zeCommandListAppendHostFunction = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(
      loadZeExtensionFunction(ZeDriver, "zeCommandListAppendHostFunction",
                              zeCommandListAppendHostFunction));

  // Allocate shared memory (accessible from both device and host)
  uint32_t *DataShared = malloc_shared<uint32_t>(N, Queue);

  // Get the command list before recording starts
  ze_command_list_handle_t ZeCommandList;
  bool success = getCommandListFromQueue(Queue, ZeCommandList);
  assert(success);

  exp_ext::command_graph Graph{
      Context, Device, {exp_ext::property::graph::enable_native_recording{}}};

  CommandListStateVerifier verifier(ZeCommandList);
  verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);
  verifier.verify(RECORDING);

  // 1. Record SYCL kernel - initialize data
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { DataShared[idx] = idx + 10; });
  });

  // 2. Record SYCL kernel - multiply by 2
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { DataShared[idx] = DataShared[idx] * 2; });
  });

  // 3. Record L0 host function directly to the recording command list
  // The host function will run on the host but is part of the command list,
  // so it will execute after the kernels complete

  ASSERT_ZE_RESULT_SUCCESS(zeCommandListAppendHostFunction(
      ZeCommandList, reinterpret_cast<void *>(HostFunction),
      static_cast<void *>(DataShared), nullptr, nullptr, 0, nullptr));

  Graph.end_recording(Queue);
  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  // Verify results
  // SYCL kernel: Data = (i + 10) * 2
  // Host function: Data = (i + 10) * 2 * 3
  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = (i + 10) * 2 * 3;
    assert(check_value(i, Expected, DataShared[i], "DataShared"));
  }

  free(DataShared, Queue);
  return 0;
}
