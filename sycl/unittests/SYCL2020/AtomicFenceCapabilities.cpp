//==----------- AtomicFenceCapabilities.cpp --- queue unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

using namespace sycl;

namespace {

thread_local bool deviceGetInfoCalled;

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

pi_result redefinedDeviceGetInfoAfter(pi_device device,
                                      pi_device_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  if (param_name == PI_EXT_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (param_value) {
      auto *Result =
          reinterpret_cast<pi_memory_order_capabilities *>(param_value);
      if (device == reinterpret_cast<pi_device>(1)) {
        std::cout << "Order Device 1" << std::endl;
        *Result = PI_MEMORY_ORDER_RELAXED | PI_MEMORY_ORDER_ACQUIRE |
                  PI_MEMORY_ORDER_RELEASE | PI_MEMORY_ORDER_ACQ_REL |
                  PI_MEMORY_ORDER_SEQ_CST;
      }
      if (device == reinterpret_cast<pi_device>(2)) {
        std::cout << "Order Device 2" << std::endl;
        *Result = PI_MEMORY_ORDER_RELAXED | PI_MEMORY_ORDER_SEQ_CST;
      }
    }
  } else if (param_name == PI_EXT_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (param_value) {
      auto *Result =
          reinterpret_cast<pi_memory_scope_capabilities *>(param_value);
      if (device == reinterpret_cast<pi_device>(1)) {
        std::cout << "Scope Device 1" << std::endl;
        *Result = PI_MEMORY_SCOPE_WORK_ITEM | PI_MEMORY_SCOPE_SUB_GROUP |
                  PI_MEMORY_SCOPE_WORK_GROUP | PI_MEMORY_SCOPE_DEVICE |
                  PI_MEMORY_SCOPE_SYSTEM;
      }
      if (device == reinterpret_cast<pi_device>(2)) {
        std::cout << "Scope Device 2" << std::endl;
        *Result = PI_MEMORY_SCOPE_WORK_ITEM | PI_MEMORY_SCOPE_SYSTEM;
      }
    }
  }
  return PI_SUCCESS;
}

TEST(AtomicFenceCapabilitiesCheck, CheckDeviceAtomicFenceOrderCapabilities) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  device Dev = Plt.get_devices()[0];

  deviceGetInfoCalled = false;

  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  auto order_capabilities =
      Dev.get_info<sycl::info::device::atomic_fence_order_capabilities>();
  EXPECT_TRUE(deviceGetInfoCalled);
  size_t expectedSize = 5;
  EXPECT_EQ(order_capabilities.size(), expectedSize);

  auto res = std::find(order_capabilities.begin(), order_capabilities.end(),
                       sycl::memory_order::relaxed);
  EXPECT_FALSE(res == order_capabilities.end());
  res = std::find(order_capabilities.begin(), order_capabilities.end(),
                  sycl::memory_order::acquire);
  EXPECT_FALSE(res == order_capabilities.end());
  res = std::find(order_capabilities.begin(), order_capabilities.end(),
                  sycl::memory_order::release);
  EXPECT_FALSE(res == order_capabilities.end());
  res = std::find(order_capabilities.begin(), order_capabilities.end(),
                  sycl::memory_order::acq_rel);
  EXPECT_FALSE(res == order_capabilities.end());
  res = std::find(order_capabilities.begin(), order_capabilities.end(),
                  sycl::memory_order::seq_cst);
  EXPECT_FALSE(res == order_capabilities.end());
}

TEST(AtomicFenceCapabilitiesCheck, CheckDeviceAtomicFenceScopeCapabilities) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  device Dev = Plt.get_devices()[0];

  deviceGetInfoCalled = false;

  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  auto scope_capabilities =
      Dev.get_info<sycl::info::device::atomic_fence_scope_capabilities>();
  EXPECT_TRUE(deviceGetInfoCalled);
  size_t expectedSize = 5;
  EXPECT_EQ(scope_capabilities.size(), expectedSize);

  auto res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                       sycl::memory_scope::work_item);
  EXPECT_FALSE(res == scope_capabilities.end());
  res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                  sycl::memory_scope::sub_group);
  EXPECT_FALSE(res == scope_capabilities.end());
  res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                  sycl::memory_scope::work_group);
  EXPECT_FALSE(res == scope_capabilities.end());
  res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                  sycl::memory_scope::device);
  EXPECT_FALSE(res == scope_capabilities.end());
  res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                  sycl::memory_scope::system);
  EXPECT_FALSE(res == scope_capabilities.end());
}

TEST(AtomicFenceCapabilitiesCheck, CheckContextAtomicFenceOrderCapabilities) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  Mock.redefineAfter<detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);
  auto devices = Plt.get_devices();
  context Ctx{devices};

  deviceGetInfoCalled = false;
  auto order_capabilities =
      Ctx.get_info<sycl::info::context::atomic_fence_order_capabilities>();
  EXPECT_TRUE(deviceGetInfoCalled);
  size_t expectedSize = 2;
  EXPECT_EQ(order_capabilities.size(), expectedSize);

  auto res = std::find(order_capabilities.begin(), order_capabilities.end(),
                       sycl::memory_order::relaxed);
  EXPECT_FALSE(res == order_capabilities.end());
  res = std::find(order_capabilities.begin(), order_capabilities.end(),
                  sycl::memory_order::seq_cst);
  EXPECT_FALSE(res == order_capabilities.end());
}

TEST(AtomicFenceCapabilitiesCheck, CheckContextAtomicFenceScopeCapabilities) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  Mock.redefineAfter<detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);
  auto devices = Plt.get_devices();
  context Ctx{devices};

  deviceGetInfoCalled = false;

  auto scope_capabilities =
      Ctx.get_info<sycl::info::context::atomic_fence_scope_capabilities>();
  EXPECT_TRUE(deviceGetInfoCalled);
  size_t expectedSize = 2;
  EXPECT_EQ(scope_capabilities.size(), expectedSize);

  auto res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                       sycl::memory_scope::work_item);
  EXPECT_FALSE(res == scope_capabilities.end());
  res = std::find(scope_capabilities.begin(), scope_capabilities.end(),
                  sycl::memory_scope::system);
  EXPECT_FALSE(res == scope_capabilities.end());
}
} // anonymous namespace
