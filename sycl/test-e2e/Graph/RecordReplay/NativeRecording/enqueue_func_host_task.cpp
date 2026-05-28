// REQUIRES: level_zero_v2_adapter
// REQUIRES: level_zero_dev_kit
// REQUIRES: arch-intel_gpu_bmg_g21
// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-INTENDED: UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP is not supported on win&gen12.

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that syclex::host_task() can be recorded into a native-recording SYCL
// Graph and executes correctly between two SYCL kernels.

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  queue Queue{property::queue::in_order{}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  uint32_t *Data = malloc_shared<uint32_t>(N, Queue);

  ze_command_list_handle_t ZeCommandList;
  bool success = getCommandListFromQueue(Queue, ZeCommandList);
  assert(success);

  exp_ext::command_graph Graph{
      Context, Device, {exp_ext::property::graph::enable_native_recording{}}};

  CommandListStateVerifier verifier(ZeCommandList);
  verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);
  verifier.verify(RECORDING);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      Data[idx] = static_cast<uint32_t>(idx[0]) + 1;
    });
  });

  syclex::host_task(Queue, [=] {
    for (size_t i = 0; i < N; i++) {
      Data[i] *= 2;
    }
  });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] += 10; });
  });

  Graph.end_recording(Queue);
  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = static_cast<uint32_t>((i + 1) * 2 + 10);
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  free(Data, Queue);
  return 0;
}
