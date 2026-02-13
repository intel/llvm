//==----------- AtomicFenceCapabilities.cpp --- queue unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

namespace {

thread_local bool deviceGetInfoCalled;

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

ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (*params.ppPropValue) {
      auto *Result = reinterpret_cast<ur_memory_order_capability_flags_t *>(
          *params.ppPropValue);
      if (*params.phDevice == reinterpret_cast<ur_device_handle_t>(1)) {
        std::cout << "Order Device 1" << std::endl;
        *Result = UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED | UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                  UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE | UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
                  UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
      if (*params.phDevice == reinterpret_cast<ur_device_handle_t>(2)) {
        std::cout << "Order Device 2" << std::endl;
        *Result = UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED | UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
      }
    }
  } else if (*params.ppropName ==
             UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (*params.ppPropValue) {
      auto *Result = reinterpret_cast<ur_memory_scope_capability_flags_t *>(
          *params.ppPropValue);
      if (*params.phDevice == reinterpret_cast<ur_device_handle_t>(1)) {
        std::cout << "Scope Device 1" << std::endl;
        *Result = UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM | UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                  UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP | UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                  UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
      if (*params.phDevice == reinterpret_cast<ur_device_handle_t>(2)) {
        std::cout << "Scope Device 2" << std::endl;
        *Result = UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM | UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;
      }
    }
  }
  return UR_RESULT_SUCCESS;
}

TEST(AtomicFenceCapabilitiesCheck, CheckDeviceAtomicFenceOrderCapabilities) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  device Dev = Plt.get_devices()[0];

  deviceGetInfoCalled = false;

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
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
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  device Dev = Plt.get_devices()[0];

  deviceGetInfoCalled = false;

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
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
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDevicesGet);
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
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDevicesGet);
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
