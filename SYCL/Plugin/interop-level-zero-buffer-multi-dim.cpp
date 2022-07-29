// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero %GPU_RUN_PLACEHOLDER %t.out

// Test 2D and 3D interoperability buffers for the Level Zero backend.

#include "interop-level-zero-buffer-helpers.hpp"
#include <sycl/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace sycl;

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  try {
    platform Plt{gpu_selector{}};

    auto Devices = Plt.get_devices();

    if (Devices.size() < 1) {
      std::cout << "Devices not found" << std::endl;
      return 0;
    }

    device Device = Devices[0];
    context Context{Device};
    queue Queue{Context, Device};

    // Get native Level Zero handles
    auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
    auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);

    ze_host_mem_alloc_desc_t HostDesc = {};
    HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    HostDesc.pNext = nullptr;
    HostDesc.flags = 0;

    ze_device_mem_alloc_desc_t DeviceDesc = {};
    DeviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    DeviceDesc.ordinal = 0;
    DeviceDesc.flags = 0;
    DeviceDesc.pNext = nullptr;

    // Test case #1
    // Check 2D buffer
    void *NativeBuffer = nullptr;
    if (is_discrete(Device))
      zeMemAllocDevice(ZeContext, &DeviceDesc, 24 * sizeof(int), 1, ZeDevice,
                       &NativeBuffer);
    else
      zeMemAllocHost(ZeContext, &HostDesc, 24 * sizeof(int), 1, &NativeBuffer);
    {
      backend_input_t<backend::ext_oneapi_level_zero, buffer<int, 1>>
          BufferInteropInput = {NativeBuffer,
                                ext::oneapi::level_zero::ownership::keep};
      auto BufferInterop = make_buffer<backend::ext_oneapi_level_zero, int, 1>(
          BufferInteropInput, Context);

      auto Buf2D = BufferInterop.reinterpret<int>(range<2>(4, 6));

      Queue.submit([&](sycl::handler &CGH) {
        auto Acc2D = Buf2D.get_access<sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel2D>([=]() {
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 6; j++) {
              Acc2D[i][j] = i + j;
            }
          }
        });
      });
      Queue.wait();

      {
        auto HostAcc2D = Buf2D.get_host_access();
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 6; j++) {
            assert(HostAcc2D[i][j] == i + j);
          }
        }
      }

      // Test case #2
      // Check 3D buffer
      auto Buf3D = BufferInterop.reinterpret<int>(range<3>(4, 2, 3));

      Queue.submit([&](sycl::handler &CGH) {
        auto Acc3D = Buf3D.get_access<sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel3D>([=]() {
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
              for (int k = 0; k < 3; k++) {
                Acc3D[i][j][k] = i + j + k;
              }
            }
          }
        });
      });
      Queue.wait();
      {
        auto HostAcc3D = Buf3D.get_host_access();
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
              assert(HostAcc3D[i][j][k] == i + j + k);
            }
          }
        }
      }
    }
    zeMemFree(ZeContext, NativeBuffer);
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
    return 0;
  }
#else
  std::cout << "Test skipped due to missing support for Level-Zero backend."
            << std::endl;
#endif
  return 0;
}
