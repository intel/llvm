//  Test that launching a kernel using level-zero interop in a graph's host_task
//  works as expected.

#include "../graph_common.hpp"
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/interop_handle.hpp>

bool getDevice(device &OutDevice, backend Backend) {
  auto Platforms = platform::get_platforms();
  platform L0Platform;
  for (auto &Platform : Platforms) {
    if (Platform.get_backend() == Backend) {
      L0Platform = Platform;
    }
  }

  auto Devices = L0Platform.get_devices();
  for (auto &Device : Devices) {
    if (Device.get_backend() == Backend) {
      OutDevice = Device;
      return true;
    }
  }
  return false;
}

std::vector<uint8_t> loadSpirvFromFile(std::string FileName) {
  std::ifstream SpvStream(FileName, std::ios::binary);
  SpvStream.seekg(0, std::ios::end);
  size_t sz = SpvStream.tellg();
  SpvStream.seekg(0);
  std::vector<uint8_t> Spv(sz);
  SpvStream.read(reinterpret_cast<char *>(Spv.data()), sz);

  return Spv;
}

int main(int, char **argv) {

  device Device;
  if (!getDevice(Device, backend::ext_oneapi_level_zero)) {
    // No suitable device found.
    return 0;
  }

  std::vector<uint8_t> Spirv = loadSpirvFromFile(argv[1]);

  const sycl::context Context{Device};
  queue Queue{Context, Device};

  std::vector<uint32_t> HostZ(Size);
  std::vector<uint32_t> HostX(Size);
  std::vector<uint32_t> ReferenceZ(Size);
  std::vector<uint32_t> ReferenceX(Size);

  std::iota(HostZ.begin(), HostZ.end(), 1);
  std::iota(HostX.begin(), HostX.end(), 10);

  for (int i = 0; i < Size; ++i) {
    ReferenceZ[i] = HostX[i] * 2 + HostZ[i];
    ReferenceX[i] = HostX[i];
  }

  uint32_t *MemZ = malloc_device<uint32_t>(Size, Queue);
  uint32_t *MemX = malloc_device<uint32_t>(Size, Queue);

  exp_ext::command_graph Graph{Context, Device};

  auto NodeA = add_node(
      Graph, Queue, [&](handler &CGH) { CGH.copy(HostZ.data(), MemZ, Size); });

  auto NodeB = add_node(
      Graph, Queue, [&](handler &CGH) { CGH.copy(HostX.data(), MemX, Size); });

  auto NodeC = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, {NodeA, NodeB});
        CGH.host_task([&]() {
          auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
          auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);

          ze_result_t status;
          ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                         nullptr,
                                         ZE_MODULE_FORMAT_IL_SPIRV,
                                         Spirv.size(),
                                         Spirv.data(),
                                         nullptr,
                                         nullptr};
          ze_module_handle_t ZeModule;
          status = zeModuleCreate(ZeContext, ZeDevice, &moduleDesc, &ZeModule,
                                  nullptr);
          assert(status == ZE_RESULT_SUCCESS);

          ze_kernel_desc_t kernelDesc = {
              ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
              "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E5saxpy"};
          ze_kernel_handle_t ZeKernel;
          status = zeKernelCreate(ZeModule, &kernelDesc, &ZeKernel);
          assert(status == ZE_RESULT_SUCCESS);

          auto ZeCommandQueueDesc =
              ze_command_queue_desc_t{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                      nullptr,
                                      0,
                                      0,
                                      0,
                                      ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

          ze_command_list_handle_t ZeCommandList;
          status = zeCommandListCreateImmediate(
              ZeContext, ZeDevice, &ZeCommandQueueDesc, &ZeCommandList);
          assert(status == ZE_RESULT_SUCCESS);

          status = zeKernelSetArgumentValue(ZeKernel, 0, sizeof(MemZ), &MemZ);
          assert(status == ZE_RESULT_SUCCESS);
          status = zeKernelSetArgumentValue(ZeKernel, 1, sizeof(MemX), &MemX);
          assert(status == ZE_RESULT_SUCCESS);

          uint32_t GroupSizeX = 32;
          uint32_t GroupSizeY = 1;
          uint32_t GroupSizeZ = 1;
          status = zeKernelSuggestGroupSize(ZeKernel, Size, 1, 1, &GroupSizeX,
                                            &GroupSizeY, &GroupSizeZ);
          assert(status == ZE_RESULT_SUCCESS);

          status = zeKernelSetGroupSize(ZeKernel, GroupSizeX, GroupSizeY,
                                        GroupSizeZ);
          assert(status == ZE_RESULT_SUCCESS);

          ze_group_count_t ZeGroupCount{
              static_cast<uint32_t>(Size) / GroupSizeX, 1, 1};
          status = zeCommandListAppendLaunchKernel(
              ZeCommandList, ZeKernel, &ZeGroupCount, nullptr, 0, nullptr);
          assert(status == ZE_RESULT_SUCCESS);

          status = zeCommandListHostSynchronize(ZeCommandList, 0);
          assert(status == ZE_RESULT_SUCCESS);

          status = zeCommandListDestroy(ZeCommandList);
          assert(status == ZE_RESULT_SUCCESS);

          status = zeKernelDestroy(ZeKernel);
          assert(status == ZE_RESULT_SUCCESS);

          status = zeModuleDestroy(ZeModule);
          assert(status == ZE_RESULT_SUCCESS);
        });
      },
      NodeA, NodeB);

  auto NodeD = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC);
        CGH.copy(MemZ, HostZ.data(), Size);
      },
      NodeC);

  auto NodeE = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeC);
        CGH.copy(MemX, HostX.data(), Size);
      },
      NodeC);

  auto GraphExec = Graph.finalize();
  Queue.ext_oneapi_graph(GraphExec);
  Queue.wait_and_throw();

  sycl::free(MemZ, Context);
  sycl::free(MemX, Context);

  for (uint32_t i = 0; i < Size; ++i) {
    assert(check_value(i, ReferenceZ[i], HostZ[i], "HostZ"));
    assert(check_value(i, ReferenceX[i], HostX[i], "HostX"));
  }

  return 0;
}
