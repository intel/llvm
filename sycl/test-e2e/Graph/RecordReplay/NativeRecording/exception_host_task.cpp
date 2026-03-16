// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{{property::queue::in_order{},
               ext::intel::property::queue::immediate_command_list{}}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  constexpr size_t N = 1024;
  int *Data = malloc_shared<int>(N, Queue);

  Graph.begin_recording(Queue);

  // Try to record a SYCL host_task - this should throw an exception
  if (!expectException(
          [&]() {
            Queue.submit([&](handler &CGH) {
              CGH.host_task([=]() {
                // This host task should not execute in native recording mode
                for (size_t i = 0; i < N; i++) {
                  Data[i] = i + 100;
                }
              });
            });
          },
          "host_task in native recording")) {
    Graph.end_recording();
    free(Data, Queue);
    return 1;
  }

  Graph.end_recording();
  free(Data, Queue);

  return 0;
}
