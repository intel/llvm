// REQUIRES: level_zero_v2_adapter && level_zero_dev_kit && arch-intel_gpu_bmg_g21
// REQUIRES-INTEL-DRIVER: lin: 37561, win: 101.8724
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out | FileCheck %s
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Some libraries allocate resources for the graph which are tied to its
// lifetime. For libraries directly writing L0, zeCommandListGetGraphExp +
// zeGraphSetDestructionCallbackExp is the standard way to do it. This test
// validates that use case with an allocation used during graph execution.

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/properties/all_properties.hpp>

struct CallbackData {
  void *allocated_memory;
  sycl::context *sycl_context;
};

void HOST_ONLY_ZE_CALLBACK printCallback(void *pUserData) {
  std::cout << "CALLBACK_INVOKED" << std::endl;
}

void HOST_ONLY_ZE_CALLBACK memoryCleanupCallback(void *pUserData) {
  CallbackData *data = static_cast<CallbackData *>(pUserData);
  if (data && data->allocated_memory && data->sycl_context) {
    sycl::free(data->allocated_memory, *(data->sycl_context));
    data->allocated_memory = nullptr;
  }
}

int main() {
  queue Queue{property::queue::in_order{}};
  auto Context = Queue.get_context();
  auto Device = Queue.get_device();

  exp_ext::command_graph Graph{
      Context, Device, {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);

  ze_driver_handle_t ZeDriver = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(getDriver(ZeDriver));

  ze_command_list_handle_t ZeCommandList = nullptr;
  assert(getCommandListFromQueue(Queue, ZeCommandList));

  CommandListStateVerifier verifier(ZeCommandList);
  verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);
  verifier.verify(RECORDING);

  zeCommandListGetGraphExp_fn GetGraph = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(
      loadZeExtensionFunction(ZeDriver, "zeCommandListGetGraphExp", GetGraph));

  ze_graph_handle_t ModifiableHandle = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(GetGraph(ZeCommandList, &ModifiableHandle));
  assert(ModifiableHandle != nullptr);

  zeGraphSetDestructionCallbackExp_fn SetDestructionCallback = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(loadZeExtensionFunction(
      ZeDriver, "zeGraphSetDestructionCallbackExp", SetDestructionCallback));

  CallbackData CbData = {Data, &Context};
  ASSERT_ZE_RESULT_SUCCESS(SetDestructionCallback(
      ModifiableHandle, printCallback, nullptr, nullptr));
  ASSERT_ZE_RESULT_SUCCESS(SetDestructionCallback(
      ModifiableHandle, memoryCleanupCallback, &CbData, nullptr));

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  });

  Graph.end_recording(Queue);
  verifier.verify(EXECUTING);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait();

  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == static_cast<int>(i));
  }

  assert(CbData.allocated_memory != nullptr && "Memory should still be alive");
  std::cout << "BEFORE_GRAPH_DESTRUCTION" << std::endl;
  {
    [[maybe_unused]] auto tmp1 = std::move(ExecGraph);
    [[maybe_unused]] auto tmp2 = std::move(Graph);
  }
  std::cout << "AFTER_GRAPH_DESTRUCTION" << std::endl;
  assert(CbData.allocated_memory == nullptr && "Memory should have been freed");

  return 0;
}
// CHECK: BEFORE_GRAPH_DESTRUCTION
// CHECK: CALLBACK_INVOKED
// CHECK-NOT: CALLBACK_INVOKED
// CHECK: AFTER_GRAPH_DESTRUCTION
