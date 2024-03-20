// REQUIRES: level_zero, level_zero_dev_kit
// L0 plugin incorrectly reports memory leaks because it doesn't take into
// account direct calls to L0 API.
// UNSUPPORTED: ze_debug
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out

// Test interoperability buffer for the Level Zer backend

#include <iostream>
#include <sycl/detail/core.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace sycl;

class DiscreteSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &Device) const final {
    if (!Device.is_gpu() ||
        Device.get_backend() != backend::ext_oneapi_level_zero)
      return -1;
    auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);
    ze_device_properties_t ZeDeviceProps;
    ZeDeviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZeDeviceProps.pNext = nullptr;
    zeDeviceGetProperties(ZeDevice, &ZeDeviceProps);
    if (!(ZeDeviceProps.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED))
      return 100;
    return -1;
  }
};

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  try {
    queue Queue{};

    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    // Get native Level Zero handles
    auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
    auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);

    ze_host_mem_alloc_desc_t HostDesc = {};
    HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    HostDesc.pNext = nullptr;
    HostDesc.flags = 0;

    // Test case #1
    // Check API
    void *HostBuffer1 = nullptr;
    zeMemAllocHost(ZeContext, &HostDesc, 10, 1, &HostBuffer1);

    backend_input_t<backend::ext_oneapi_level_zero, buffer<char, 1>>
        HostBufferInteropInput1 = {
            HostBuffer1, ext::oneapi::level_zero::ownership::transfer};
    auto HostBufferInterop1 =
        make_buffer<backend::ext_oneapi_level_zero, char, 1>(
            HostBufferInteropInput1, Context);

    auto Event = Queue.single_task([=]() {});

    void *HostBuffer2 = nullptr;
    zeMemAllocHost(ZeContext, &HostDesc, 12 * sizeof(int), 1, &HostBuffer2);
    backend_input_t<backend::ext_oneapi_level_zero, buffer<int, 1>>
        HostBufferInteropInput2 = {
            HostBuffer2, ext::oneapi::level_zero::ownership::transfer};
    auto HostBufferInterop2 =
        make_buffer<backend::ext_oneapi_level_zero, int, 1>(
            HostBufferInteropInput2, Context, Event);

    Queue.submit([&](sycl::handler &CGH) {
      auto Acc1 =
          HostBufferInterop1.get_access<sycl::access::mode::read_write>(CGH);
      auto Acc2 = HostBufferInterop2.get_access<sycl::access::mode::read_write>(
          CGH, range<1>(12));

      CGH.single_task<class SimpleKernel1>([=]() {
        for (int i = 0; i < 10; i++) {
          Acc1[i] = 'a';
        }

        for (int i = 0; i < 12; i++) {
          Acc2[i] = 10;
        }
      });
    });
    Queue.wait();

    {
      auto HostAcc1 = HostBufferInterop1.get_host_access();
      for (int i = 0; i < 10; i++) {
        assert(HostAcc1[i] == 'a');
      }

      auto HostAcc2 = HostBufferInterop2.get_host_access();
      for (int i = 0; i < 12; i++) {
        assert(HostAcc2[i] == 10);
      }
    }

    // Test case #2
    // Check sub-buffer creation
    auto SubBuffer1 = buffer(HostBufferInterop2, id<1>(0), range<1>(3));
    auto SubBuffer2 = buffer(HostBufferInterop2, id<1>(3), range<1>(9));

    Queue.submit([&](sycl::handler &CGH) {
      auto Acc1 = SubBuffer1.get_access<sycl::access::mode::read_write>(CGH);
      auto Acc2 = SubBuffer2.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel2>([=]() {
        for (int i = 0; i < 3; i++) {
          Acc1[i] = 77;
        }
        for (int i = 0; i < 9; i++) {
          Acc2[i] = 99;
        }
      });
    });
    Queue.wait();

    {
      auto HostAcc1 = SubBuffer1.get_host_access();
      for (int i = 0; i < 3; i++) {
        assert(HostAcc1[i] == 77);
      }
    }
    {
      auto HostAcc2 = SubBuffer2.get_host_access();

      for (int i = 0; i < 9; i++) {
        assert(HostAcc2[i] == 99);
      }
    }

    // Test case #3
    // Use buffer in two different contexts
    context Context1;
    queue Queue1(Context1, default_selector_v);
    Queue1.submit([&](sycl::handler &CGH) {
      auto Acc =
          HostBufferInterop2.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel3>([=]() {
        for (int i = 0; i < 12; i++) {
          Acc[i] = 99;
        }
      });
    });

    context Context2;
    queue Queue2(Context2, default_selector_v);
    Queue2.submit([&](sycl::handler &CGH) {
      auto Acc =
          HostBufferInterop2.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel4>([=]() {
        for (int i = 0; i < 12; i++) {
          Acc[i] *= 2;
        }
      });
    });

    {
      auto HostAcc = HostBufferInterop2.get_host_access();

      for (int i = 0; i < 12; i++) {
        assert(HostAcc[i] == 198);
      }
    }
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
    return 0;
  }

  try {
    // Test case #4
    // Check device/shared allocations and host accessor.
    platform Plt{DiscreteSelector{}};

    auto Devices = Plt.get_devices();

    if (Devices.size() <= 1)
      return 0;

    device Dev = Devices[0];
    context Context{Dev};
    queue Queue{Context, Dev};

    auto Device = Queue.get_info<info::queue::device>();

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

    void *SharedBuffer = nullptr;
    zeMemAllocShared(ZeContext, &DeviceDesc, &HostDesc, 12 * sizeof(int), 1,
                     nullptr, &SharedBuffer);

    backend_input_t<backend::ext_oneapi_level_zero, buffer<int, 1>>
        SharedBufferInteropInput = {
            SharedBuffer, ext::oneapi::level_zero::ownership::transfer};
    auto SharedBufferInterop =
        make_buffer<backend::ext_oneapi_level_zero, int, 1>(
            SharedBufferInteropInput, Context);

    void *DeviceBuffer = nullptr;
    zeMemAllocDevice(ZeContext, &DeviceDesc, 12 * sizeof(int), 1, ZeDevice,
                     &DeviceBuffer);

    backend_input_t<backend::ext_oneapi_level_zero, buffer<int, 1>>
        DeviceBufferInteropInput = {
            DeviceBuffer, ext::oneapi::level_zero::ownership::transfer};
    auto DeviceBufferInterop =
        make_buffer<backend::ext_oneapi_level_zero, int, 1>(
            DeviceBufferInteropInput, Context);

    Queue.submit([&](sycl::handler &CGH) {
      auto Acc1 =
          SharedBufferInterop.get_access<sycl::access::mode::read_write>(CGH);
      auto Acc2 =
          DeviceBufferInterop.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel5>([=]() {
        for (int i = 0; i < 12; i++) {
          Acc1[i] = 77;
        }
        for (int i = 0; i < 12; i++) {
          Acc2[i] = 99;
        }
      });
    });
    Queue.wait();
    {
      auto HostAcc1 = SharedBufferInterop.get_host_access();
      for (int i = 0; i < 12; i++) {
        assert(HostAcc1[i] == 77);
      }
      auto HostAcc2 = DeviceBufferInterop.get_host_access();
      for (int i = 0; i < 12; i++) {
        assert(HostAcc2[i] == 99);
      }
    }

    // Test case #5
    // Use device buffer in two different contexts
    Queue.submit([&](sycl::handler &CGH) {
      auto Acc =
          DeviceBufferInterop.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel6>([=]() {
        for (int i = 0; i < 12; i++) {
          Acc[i] = 99;
        }
      });
    });

    // Submit in a different context with one device
    device Dev2 = Devices.size() > 1 ? Devices[1] : Devices[0];
    context Context2{Dev2};
    queue Queue2{Context2, Dev2};
    Queue2.submit([&](sycl::handler &CGH) {
      auto Acc =
          DeviceBufferInterop.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel7>([=]() {
        for (int i = 0; i < 12; i++) {
          Acc[i] *= 2;
        }
      });
    });

    // Submit in a different context with possibly multiple device
    queue Queue3;
    Queue3.submit([&](sycl::handler &CGH) {
      auto Acc =
          DeviceBufferInterop.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel8>([=]() {
        for (int i = 0; i < 12; i++) {
          Acc[i] += 2;
        }
      });
    });

    {
      auto HostAcc = DeviceBufferInterop.get_host_access();

      for (int i = 0; i < 12; i++) {
        assert(HostAcc[i] == 200);
      }
    }
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
