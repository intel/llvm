//==-------- AtomicMemoryScopeCapabilities.cpp --- queue unit tests --------==//
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

pi_result redefinedDeviceGetInfoAfter(pi_device device,
                                      pi_device_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES) {
    deviceGetInfoCalled = true;
    if (param_value) {
      auto *Result =
          reinterpret_cast<pi_memory_scope_capabilities *>(param_value);
      *Result = PI_MEMORY_SCOPE_WORK_ITEM | PI_MEMORY_SCOPE_SUB_GROUP |
                PI_MEMORY_SCOPE_WORK_GROUP | PI_MEMORY_SCOPE_DEVICE |
                PI_MEMORY_SCOPE_SYSTEM;
    }
  }
  return PI_SUCCESS;
}

TEST(AtomicMemoryScopeCapabilitiesCheck, CheckAtomicMemoryScopeCapabilities) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  device Dev = Plt.get_devices()[0];

  deviceGetInfoCalled = false;

  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  auto scope_capabilities =
      Dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
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
} // anonymous namespace
