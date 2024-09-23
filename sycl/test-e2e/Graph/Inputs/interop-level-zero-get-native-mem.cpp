// Tests interop with host_task in graph
// This test was taken from
// `sycl/test-e2e/Plugin/interop-level-zero-get-native-mem.cpp` This test has
// been simplified to only work with signle device.

#include "../graph_common.hpp"
// Level-Zero
#include <level_zero/ze_api.h>
// SYCL
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <sycl/interop_handle.hpp>

bool is_discrete(const device &Device) {
  auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);
  ze_device_properties_t ZeDeviceProps;
  ZeDeviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ZeDeviceProps.pNext = nullptr;
  zeDeviceGetProperties(ZeDevice, &ZeDeviceProps);
  return !(ZeDeviceProps.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
}

int main() {
  try {
    platform Plt{gpu_selector_v};

    auto Devices = Plt.get_devices();

    if (Devices.size() < 1) {
      std::cout << "Devices not found" << std::endl;
      return 0;
    }

    device Dev1 = Devices[0];
    context Context1{Dev1};
    queue Queue{Context1, Dev1};

    auto Context = Queue.get_context();
    auto Device = Queue.get_info<info::queue::device>();

    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Get native Level Zero handles
    auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
    auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);

    ze_device_mem_alloc_desc_t DeviceDesc = {};
    DeviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    DeviceDesc.ordinal = 0;
    DeviceDesc.flags = 0;
    DeviceDesc.pNext = nullptr;

    ze_host_mem_alloc_desc_t HostDesc = {};
    HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    HostDesc.pNext = nullptr;
    HostDesc.flags = 0;

    void *NativeBuffer = nullptr;
    if (is_discrete(Device))
      // Use shared allocation (the check contents on the host later) associated
      // with a device if the device is discreet.
      zeMemAllocShared(ZeContext, &DeviceDesc, &HostDesc, 12 * sizeof(int), 1,
                       ZeDevice, &NativeBuffer);
    else
      // Use host allocation if device is integrated.
      zeMemAllocHost(ZeContext, &HostDesc, 12 * sizeof(int), 1, &NativeBuffer);

    backend_input_t<backend::ext_oneapi_level_zero, buffer<int, 1>>
        BufferInteropInput = {NativeBuffer,
                              ext::oneapi::level_zero::ownership::transfer};
    {
      auto BufferInterop = make_buffer<backend::ext_oneapi_level_zero, int, 1>(
          BufferInteropInput, Context);

      auto NodeA = add_node(Graph, Queue, [&](sycl::handler &CGH) {
        auto Acc =
            BufferInterop.get_access<sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel6>([=]() {
          for (int i = 0; i < 12; i++) {
            Acc[i] = 99;
          }
        });
      });

      auto NodeB = add_node(
          Graph, Queue,
          [&](sycl::handler &CGH) {
            depends_on_helper(CGH, NodeA);
            auto Acc =
                BufferInterop.get_access<sycl::access::mode::read_write>(CGH);
            CGH.single_task<class SimpleKernel7>([=]() {
              for (int i = 0; i < 12; i++) {
                Acc[i] *= 2;
              }
            });
          },
          NodeA);

      add_node(
          Graph, Queue,
          [&](handler &CGH) {
            depends_on_helper(CGH, NodeB);
            auto BufferAcc = BufferInterop.get_access<access::mode::write>(CGH);
            CGH.host_task([=](const interop_handle &IH) {
              void *DevicePtr =
                  IH.get_native_mem<backend::ext_oneapi_level_zero>(BufferAcc);
              ze_memory_allocation_properties_t MemAllocProperties{};
              ze_result_t Res = zeMemGetAllocProperties(
                  ZeContext, DevicePtr, &MemAllocProperties, nullptr);
              assert(Res == ZE_RESULT_SUCCESS);

              assert(NativeBuffer == DevicePtr);

              int *CastedPtr = (int *)DevicePtr;
              for (int i = 0; i < 12; i++)
                assert(CastedPtr[i] == 198);
            });
          },
          NodeB);

      auto GraphExec = Graph.finalize();

      auto Event =
          Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
      Queue.wait_and_throw();
    }

  } catch (exception &e) {
    std::cout << e.what() << std::endl;
  }
  return 0;
}
