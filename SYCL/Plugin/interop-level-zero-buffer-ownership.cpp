// REQUIRES: gpu, level_zero, level_zero_dev_kit
// L0 plugin incorrectly reports memory leaks because it doesn't take into
// account direct calls to L0 API.
// UNSUPPORTED: ze_debug-1,ze_debug4
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1 SYCL_DEVICE_FILTER=level_zero ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck %s

// Test for Level Zero buffer interop API.
// Check the following cases:
// 1. User-provided memory allocation is not freed by DPCPP RT if
// "keep" ownership is specified.
// 2. User-provided memory allocation is freed by DPCPP RT if
// "transfer" ownership is specified.

// NOTE: SYCL RT will see unbalanced count of alloc/free,
// so this test will fail with ZE_DEBUG=4.

// Keep ownership
// CHECK: zeMemFree

// Transfer ownership
// CHECK: zeMemFree
// CHECK: zeMemFree

// No other calls to zeMemFree
// CHECK-NOT: zeMemFree

#include "interop-level-zero-buffer-helpers.hpp"
#include <sycl/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace cl::sycl;

// Test copy back depending on provided ownership and check that memory is freed
// properly.
void test_copyback_and_free(
    queue &Queue1, queue &Queue2,
    const ext::oneapi::level_zero::ownership &Ownership) {
  try {
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
        BufferInteropInput = {NativeBuffer, Ownership};
    {
      auto BufferInterop = make_buffer<backend::ext_oneapi_level_zero, int, 1>(
          BufferInteropInput, Context);

      auto Event = Queue1.submit([&](cl::sycl::handler &CGH) {
        auto Acc =
            BufferInterop.get_access<cl::sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel6>([=]() {
          for (int i = 0; i < 12; i++) {
            Acc[i] = 99;
          }
        });
      });
      Event.wait();

      // Submit in a different context
      Queue2.submit([&](cl::sycl::handler &CGH) {
        auto Acc =
            BufferInterop.get_access<cl::sycl::access::mode::read_write>(CGH);
        CGH.single_task<class SimpleKernel7>([=]() {
          for (int i = 0; i < 12; i++) {
            Acc[i] *= 2;
          }
        });
      });

      Queue2.wait();
    }
    if (Ownership == ext::oneapi::level_zero::ownership::keep)
      zeMemFree(ZeContext, NativeBuffer);
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  try {
    platform Plt{gpu_selector{}};

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

    std::cout << "Test case #1: Keep ownership" << std::endl;
    test_copyback_and_free(Queue1, Queue2,
                           ext::oneapi::level_zero::ownership::keep);

    std::cout << "Test case #2: Transfer ownership" << std::endl;
    test_copyback_and_free(Queue1, Queue2,
                           ext::oneapi::level_zero::ownership::transfer);

  } catch (exception &e) {
    std::cout << e.what() << std::endl;
  }
#else
  std::cout << "Test skipped due to missing support for Level-Zero backend."
            << std::endl;
#endif
  return 0;
}
