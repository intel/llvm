//==---- AtomicMemoryOrderCapabilities.cpp --- memory order query test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

namespace {

thread_local bool deviceGetInfoCalled;

static bool has_capability(const std::vector<memory_order> &deviceCapabilities,
                           memory_order capabilityToFind) {
  return std::find(deviceCapabilities.begin(), deviceCapabilities.end(),
                   capabilityToFind) != deviceCapabilities.end();
}

ur_result_t redefinedDevicesGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries > 0) {
    (*params.pphDevices)[0] = reinterpret_cast<ur_device_handle_t>(1);
    (*params.pphDevices)[1] = reinterpret_cast<ur_device_handle_t>(2);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (*params.ppPropValue) {
      ur_memory_order_capability_flags_t *Capabilities =
          reinterpret_cast<ur_memory_order_capability_flags_t *>(
              *params.ppPropValue);
      if (*params.phDevice == reinterpret_cast<ur_device_handle_t>(1)) {
        *Capabilities = UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED | UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE | UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
                        UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
      if (*params.phDevice == reinterpret_cast<ur_device_handle_t>(2)) {
        *Capabilities = UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED | UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
    }
  }
  return UR_RESULT_SUCCESS;
}

TEST(AtomicMemoryOrderCapabilities, DeviceQueryReturnsCorrectCapabilities) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfo);

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
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDevicesGet);

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
