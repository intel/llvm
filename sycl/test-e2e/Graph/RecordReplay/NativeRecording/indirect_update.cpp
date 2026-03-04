// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// A pointer to a structure may be utilized to hold mutable kernel arguments
// without using mutable command lists as the pointer itself remains constant.
// This test demonstrates updating input/output pointers and scalar arguments
// between graph submissions with a single wait at the end.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

struct MutableArguments {
  int *InputPtr;
  int *OutputPtr;
  int Multiplier;
  int Addend;
};

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 1024;

  int *Input1 = malloc_device<int>(N, Queue);
  int *Input2 = malloc_device<int>(N, Queue);
  int *Output1 = malloc_device<int>(N, Queue);
  int *Output2 = malloc_device<int>(N, Queue);

  MutableArguments *DeviceArgs = malloc_device<MutableArguments>(1, Queue);

  std::vector<int> HostInput1(N);
  for (size_t i = 0; i < N; i++) {
    HostInput1[i] = static_cast<int>(i);
  }
  Queue.memcpy(Input1, HostInput1.data(), N * sizeof(int)).wait();

  std::vector<int> HostInput2(N);
  for (size_t i = 0; i < N; i++) {
    HostInput2[i] = static_cast<int>(i * 2);
  }
  Queue.memcpy(Input2, HostInput2.data(), N * sizeof(int)).wait();

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      int input = DeviceArgs->InputPtr[idx];
      int multiplier = DeviceArgs->Multiplier;
      int addend = DeviceArgs->Addend;
      DeviceArgs->OutputPtr[idx] = input * multiplier + addend;
    });
  });

  Graph.end_recording(Queue);

  auto ExecutableGraph = Graph.finalize();

  MutableArguments HostArgs1;
  HostArgs1.InputPtr = Input1;
  HostArgs1.OutputPtr = Output1;
  HostArgs1.Multiplier = 2;
  HostArgs1.Addend = 10;

  MutableArguments HostArgs2;
  HostArgs2.InputPtr = Input2;
  HostArgs2.OutputPtr = Output2;
  HostArgs2.Multiplier = 3;
  HostArgs2.Addend = 20;

  // Submit graph with first set of arguments (Input1 -> Output1)
  Queue.memcpy(DeviceArgs, &HostArgs1, sizeof(MutableArguments)).wait();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

  // Update the device arguments via memcpy
  Queue.memcpy(DeviceArgs, &HostArgs2, sizeof(MutableArguments));

  // Submit graph again with second set of arguments (Input2 -> Output2)
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

  // Single wait for both graph submissions
  Queue.wait();

  // Read back both output buffers
  std::vector<int> HostOutput1(N);
  std::vector<int> HostOutput2(N);
  Queue.memcpy(HostOutput1.data(), Output1, N * sizeof(int)).wait();
  Queue.memcpy(HostOutput2.data(), Output2, N * sizeof(int)).wait();

  // Verify first execution results:
  // Output1[i] = Input1[i] * 2 + 10 = i * 2 + 10
  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i) * 2 + 10;
    assert(check_value(i, Expected, HostOutput1[i], "Output1"));
  }

  // Verify second execution results:
  // Output2[i] = Input2[i] * 3 + 20 = (i * 2) * 3 + 20
  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i) * 6 + 20;
    assert(check_value(i, Expected, HostOutput2[i], "Output2"));
  }

  free(Input1, Queue);
  free(Input2, Queue);
  free(Output1, Queue);
  free(Output2, Queue);
  free(DeviceArgs, Queue);

  return 0;
}
