// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -lze_loader -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out %S/../../Inputs/Kernels/saxpy.spv
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out %S/../../Inputs/Kernels/saxpy.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test native recording with intermixed SYCL and Level-Zero kernels.

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"

#include <level_zero/ze_api.h>
#include <sycl/properties/all_properties.hpp>

int main(int, char **argv) {
  queue Queue{property::queue::in_order{}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  const size_t N = 1024;
  uint32_t *DataX = malloc_device<uint32_t>(N, Queue);
  uint32_t *DataZ = malloc_device<uint32_t>(N, Queue);

  std::vector<uint32_t> HostX(N);
  std::vector<uint32_t> HostZ(N);
  for (size_t i = 0; i < N; i++) {
    HostX[i] = i + 10;
    HostZ[i] = i + 1;
  }

  Queue.memcpy(DataX, HostX.data(), N * sizeof(uint32_t)).wait();
  Queue.memcpy(DataZ, HostZ.data(), N * sizeof(uint32_t)).wait();

  ZeKernelFactory KernelFactory(Queue);
  ze_module_handle_t ZeModule =
      KernelFactory.createModule(loadSpirvFromFile(argv[1]));
  ze_kernel_handle_t ZeKernel = KernelFactory.createKernel(
      ZeModule, "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E5saxpy");

  // Create graph for native recording
  exp_ext::command_graph Graph{Context, Device};

  ze_command_list_handle_t ZeCommandList;
  bool result = getCommandListFromQueue(Queue, ZeCommandList);
  assert(result);

  CommandListStateVerifier verifier(ZeCommandList);
  verifier.verify(EXECUTING);

  // Begin recording
  Graph.begin_recording(Queue);
  verifier.verify(RECORDING);

  // 1. Record SYCL kernel - multiply X by 2
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { DataX[idx] = DataX[idx] * 2; });
  });

  // 2. Record L0 kernel directly to the recording command list

  // Suggest and prepare group size
  uint32_t GroupSizeX;
  uint32_t GroupSizeY;
  uint32_t GroupSizeZ;

  ASSERT_ZE_RESULT_SUCCESS(zeKernelSuggestGroupSize(
      ZeKernel, N, 1, 1, &GroupSizeX, &GroupSizeY, &GroupSizeZ));

  // Prepare kernel arguments using the WithArguments API
  // saxpy computes Z = X * 2 + Z
  // Arguments are passed as an array of pointers
  void *ArgPointers[] = {&DataZ, &DataX};

  ze_group_count_t ZeGroupCount{static_cast<uint32_t>(N) / GroupSizeX, 1, 1};
  ze_group_size_t ZeGroupSize{GroupSizeX, GroupSizeY, GroupSizeZ};

  ASSERT_ZE_RESULT_SUCCESS(zeCommandListAppendLaunchKernelWithArguments(
      ZeCommandList, ZeKernel, ZeGroupCount, ZeGroupSize, ArgPointers, nullptr,
      nullptr, 0, nullptr));

  Graph.end_recording(Queue);
  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  Queue.memcpy(HostX.data(), DataX, N * sizeof(uint32_t)).wait();
  Queue.memcpy(HostZ.data(), DataZ, N * sizeof(uint32_t)).wait();

  // Verify results
  // SYCL kernel: X = (i + 10) * 2
  // L0 saxpy kernel: Z = X * 2 + Z = (i + 10) * 2 * 2 + (i + 1)
  for (size_t i = 0; i < N; i++) {
    uint32_t ExpectedX = (i + 10) * 2;
    uint32_t ExpectedZ = (i + 10) * 2 * 2 + (i + 1);
    assert(check_value(i, ExpectedX, HostX[i], "HostX"));
    assert(check_value(i, ExpectedZ, HostZ[i], "HostZ"));
  }

  free(DataX, Queue);
  free(DataZ, Queue);

  return 0;
}
