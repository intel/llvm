// REQUIRES: gpu,level_zero,level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// Because we initialize L0 driver directly in the test before UR L0 adapter
// does that, we need to explicitly set the right environment variables instead
// of using UR_L0_DEBUG shortcut
// RUN: env ZEL_ENABLE_LOADER_LOGGING=1 ZEL_LOADER_LOGGING_LEVEL=trace \
// RUN:     ZEL_LOADER_LOG_CONSOLE=1 ZE_ENABLE_VALIDATION_LAYER=1 \
// RUN: %{run} %t.out 2>&1 | FileCheck %s
// UNSUPPORTED: ze_debug

// L0v2 adapter doesn't optimize buffer creation based on device type yet
// (integrated buffer implementation needs more work).
// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20121

#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

using namespace sycl;

int main() {
  // Initializing Level Zero driver is required if this test is linked
  // statically with Level Zero loader, otherwise the driver will not be
  // initialized.
  ze_result_t result = zeInit(0);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed with error code: " << result << std::endl;
    return 1;
  }

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
// CHECK: [[API]](
