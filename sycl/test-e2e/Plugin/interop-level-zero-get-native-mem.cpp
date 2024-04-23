// REQUIRES: level_zero, level_zero_dev_kit
// L0 plugin incorrectly reports memory leaks because it doesn't take into
// account direct calls to L0 API.
// UNSUPPORTED: ze_debug
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Test get_native_mem for the Level Zero backend.

// Level-Zero
#include <level_zero/ze_api.h>

// SYCL
#include "interop-level-zero-buffer-helpers.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr size_t SIZE = 16;

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  try {
    platform Plt{gpu_selector_v};

    auto Devices = Plt.get_devices();

    if (Devices.size() < 1) {
      std::cout << "Devices not found" << std::endl;
      return 0;
    }

    device Dev1 = Devices[0];
    context Context1{Dev1};
    queue Queue1{Context1, Dev1};

    device Dev2 = Devices.size() > 1 ? Devices[1] : Devices[0];
    context Context2{Dev2};
    queue Queue2{Context2, Dev2};

    auto Context = Queue1.get_context();
    auto Device = Queue1.get_info<info::queue::device>();

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

      auto Event = Queue1.submit([&](sycl::handler &CGH) {
        auto Acc =
            BufferInterop.get_access<sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel6>([=]() {
          for (int i = 0; i < 12; i++) {
            Acc[i] = 99;
          }
        });
      });
      Event.wait();

      // Submit in a different context
      Queue2.submit([&](sycl::handler &CGH) {
        auto Acc =
            BufferInterop.get_access<sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel7>([=]() {
          for (int i = 0; i < 12; i++) {
            Acc[i] *= 2;
          }
        });
      });

      Queue2.wait();

      Queue1
          .submit([&](handler &CGH) {
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
          })
          .wait();
    }

  } catch (exception &e) {
    std::cout << e.what() << std::endl;
  }
#else
  std::cout << "Test skipped due to missing support for Level-Zero backend."
            << std::endl;
#endif
  return 0;
}
