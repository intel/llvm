//==-------------- DeviceInfo.cpp --- device info unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

using namespace sycl;

namespace {
struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {}

  context &Ctx;
  bool UUIDInfoCalled = false;
};
} // namespace

static std::unique_ptr<TestCtx> TestContext;

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_UUID) {
    TestContext->UUIDInfoCalled = true;
  }

  return PI_SUCCESS;
}

class DeviceInfoTest : public ::testing::Test {
public:
  DeviceInfoTest() : Plt{default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host()) {
      std::clog << "This test is only supported on non-host platforms.\n";
      std::clog << "Current platform is "
                << Plt.get_info<info::platform::name>() << "\n";
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);

    Mock->redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  }

protected:
  platform Plt;
  std::unique_ptr<unittest::PiMock> Mock;
};

TEST_F(DeviceInfoTest, GetDeviceUUID) {
  if (Plt.is_host()) {
    return;
  }

  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  device Dev = Ctx.get_devices()[0];

  if (!Dev.has(aspect::ext_intel_device_info_uuid)) {
    std::clog
        << "This test is only for the devices with UUID extension support.\n";
    return;
  }

  auto UUID = Dev.get_info<info::device::ext_intel_device_info_uuid>();

  EXPECT_EQ(TestContext->UUIDInfoCalled, true)
      << "Expect piDeviceGetInfo to be "
      << "called with PI_DEVICE_INFO_UUID";

  EXPECT_EQ(sizeof(UUID), 16 * sizeof(unsigned char))
      << "Expect device UUID to be "
      << "of 16 bytes";
}
