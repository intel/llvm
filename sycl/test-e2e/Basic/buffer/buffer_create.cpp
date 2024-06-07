// REQUIRES: gpu,level_zero,level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s
// UNSUPPORTED: ze_debug

#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

using namespace sycl;

int main() {
  constexpr int Size = 100;
  queue Queue;
  auto D = Queue.get_device();
  auto NumOfDevices = Queue.get_context().get_devices().size();
  buffer<::cl_int, 1> Buffer(Size);

  ze_device_handle_t ZeDevice =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(D);

  ze_device_properties_t ZeDeviceProperties{};
  ze_result_t ZeRes = zeDeviceGetProperties(ZeDevice, &ZeDeviceProperties);
  assert(ZeRes == ZE_RESULT_SUCCESS);

  bool IsIntegrated =
      ZeDeviceProperties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

  Queue.submit([&](handler &cgh) {
    accessor Accessor{Buffer, cgh, read_write};
    if (IsIntegrated)
      std::cerr << "Integrated GPU should use zeMemAllocHost" << std::endl;
    else
      std::cerr << "Discrete GPU should use zeMemAllocDevice" << std::endl;
    cgh.parallel_for<class CreateBuffer>(range<1>(Size),
                                         [=](id<1> ID) { Accessor[ID] = 0; });
  });
  Queue.wait();

  return 0;
}

// CHECK: {{Integrated|Discrete}} GPU should use [[API:zeMemAllocHost|zeMemAllocDevice]]
// CHECK: ZE ---> [[API]](
