//==---- AtomicMemoryOrderCapabilities.cpp --- memory order query test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

namespace {

thread_local bool deviceGetInfoCalled;

static bool has_capability(const std::vector<memory_order> &deviceCapabilities,
                           memory_order capabilityToFind) {
  return std::find(deviceCapabilities.begin(), deviceCapabilities.end(),
                   capabilityToFind) != deviceCapabilities.end();
}

pi_result redefinedDevicesGet(pi_platform platform, pi_device_type device_type,
                              pi_uint32 num_entries, pi_device *devices,
                              pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 2;
  if (devices && num_entries > 0) {
    devices[0] = reinterpret_cast<pi_device>(1);
    devices[1] = reinterpret_cast<pi_device>(2);
  }
  return PI_SUCCESS;
}

pi_result redefinedDeviceGetInfo(pi_device device, pi_device_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (param_value) {
      pi_memory_order_capabilities *Capabilities =
          reinterpret_cast<pi_memory_order_capabilities *>(param_value);
      if (device == reinterpret_cast<pi_device>(1)) {
        *Capabilities = PI_MEMORY_ORDER_RELAXED | PI_MEMORY_ORDER_ACQUIRE |
                        PI_MEMORY_ORDER_RELEASE | PI_MEMORY_ORDER_ACQ_REL |
                        PI_MEMORY_ORDER_SEQ_CST;
      }
      if (device == reinterpret_cast<pi_device>(2)) {
        *Capabilities = PI_MEMORY_ORDER_RELAXED | PI_MEMORY_ORDER_SEQ_CST;
      }
    }
  }
  return PI_SUCCESS;
}

TEST(AtomicMemoryOrderCapabilities, DeviceQueryReturnsCorrectCapabilities) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfo);

  const device Dev = Plt.get_devices()[0];
  context Ctx{Dev};

  deviceGetInfoCalled = false;

  auto Capabilities =
      Dev.get_info<info::device::atomic_memory_order_capabilities>();
  EXPECT_TRUE(deviceGetInfoCalled);
  constexpr size_t expectedCapabilityVecSize = 5;
  EXPECT_EQ(Capabilities.size(), expectedCapabilityVecSize);

  EXPECT_TRUE(has_capability(Capabilities, memory_order::relaxed));
  EXPECT_TRUE(has_capability(Capabilities, memory_order::acquire));
  EXPECT_TRUE(has_capability(Capabilities, memory_order::release));
  EXPECT_TRUE(has_capability(Capabilities, memory_order::acq_rel));
  EXPECT_TRUE(has_capability(Capabilities, memory_order::seq_cst));
}

TEST(AtomicMemoryOrderCapabilities, ContextQueryReturnsCorrectCapabilities) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfo);
  Mock.redefineAfter<detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);

  auto devices = Plt.get_devices();
  context Ctx{devices};

  deviceGetInfoCalled = false;

  auto Capabilities =
      Ctx.get_info<info::context::atomic_memory_order_capabilities>();
  EXPECT_TRUE(deviceGetInfoCalled);
  constexpr size_t expectedCapabilityVecSize = 2;
  EXPECT_EQ(Capabilities.size(), expectedCapabilityVecSize);

  EXPECT_TRUE(has_capability(Capabilities, memory_order::relaxed));
  EXPECT_TRUE(has_capability(Capabilities, memory_order::seq_cst));
}

} // namespace
