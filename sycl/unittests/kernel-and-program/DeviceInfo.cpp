//==-------------- DeviceInfo.cpp --- device info unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

namespace {
struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {}

  context &Ctx;
  bool UUIDInfoCalled = false;
  bool FreeMemoryInfoCalled = false;

  std::string BuiltInKernels;
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
  } else if (param_name == PI_DEVICE_INFO_BUILT_IN_KERNELS) {
    if (param_value_size_ret) {
      *param_value_size_ret = TestContext->BuiltInKernels.size() + 1;
    } else if (param_value) {
      char *dst = static_cast<char *>(param_value);
      dst[TestContext->BuiltInKernels.copy(dst, param_value_size)] = '\0';
    }
  } else if (param_name == PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY) {
    TestContext->FreeMemoryInfoCalled = true;
  } else if (param_name == PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE) {
    if (param_value_size_ret)
      *param_value_size_ret = 4;

    if (param_value) {
      assert(param_value_size == sizeof(uint32_t));
      *static_cast<uint32_t *>(param_value) = 800;
    }
  } else if (param_name == PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH) {
    if (param_value_size_ret)
      *param_value_size_ret = 4;

    if (param_value) {
      assert(param_value_size == sizeof(uint32_t));
      *static_cast<uint32_t *>(param_value) = 64;
    }
  }

  // This mock device has no sub-devices
  if (param_name == PI_DEVICE_INFO_PARTITION_PROPERTIES) {
    if (param_value_size_ret) {
      *param_value_size_ret = 0;
    }
  }
  if (param_name == PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    assert(param_value_size == sizeof(pi_device_affinity_domain));
    if (param_value) {
      *static_cast<pi_device_affinity_domain *>(param_value) = 0;
    }
  }

  return PI_SUCCESS;
}

class DeviceInfoTest : public ::testing::Test {
public:
  DeviceInfoTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
        redefinedDeviceGetInfo);
  }

protected:
  unittest::PiMock Mock;
  sycl::platform Plt;
};

static pi_result redefinedNegativeDeviceGetInfo(pi_device device,
                                                pi_device_info param_name,
                                                size_t param_value_size,
                                                void *param_value,
                                                size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_DEVICE_INFO_UUID:
  case PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY:
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE:
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH:
    return PI_ERROR_INVALID_VALUE;
  default:
    return PI_SUCCESS;
  }

  return PI_SUCCESS;
}

class DeviceInfoNegativeTest : public ::testing::Test {
public:
  DeviceInfoNegativeTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineBefore<detail::PiApiKind::piDeviceGetInfo>(
        redefinedNegativeDeviceGetInfo);
  }

protected:
  unittest::PiMock Mock;
  sycl::platform Plt;
};

TEST_F(DeviceInfoTest, GetDeviceUUID) {
  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  device Dev = Ctx.get_devices()[0];

  EXPECT_TRUE(Dev.has(aspect::ext_intel_device_info_uuid));

  auto UUID = Dev.get_info<ext::intel::info::device::uuid>();

  EXPECT_EQ(TestContext->UUIDInfoCalled, true)
      << "Expect piDeviceGetInfo to be "
      << "called with PI_DEVICE_INFO_UUID";

  EXPECT_EQ(sizeof(UUID), 16 * sizeof(unsigned char))
      << "Expect device UUID to be "
      << "of 16 bytes";
}

TEST_F(DeviceInfoTest, GetDeviceFreeMemory) {
  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  device Dev = Ctx.get_devices()[0];

  EXPECT_TRUE(Dev.has(aspect::ext_intel_free_memory));

  auto FreeMemory = Dev.get_info<ext::intel::info::device::free_memory>();

  EXPECT_EQ(TestContext->FreeMemoryInfoCalled, true)
      << "Expect piDeviceGetInfo to be "
      << "called with PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY";

  EXPECT_EQ(sizeof(FreeMemory), sizeof(uint64_t))
      << "Expect free_memory to be of uint64_t size";
}

TEST_F(DeviceInfoTest, GetDeviceMemoryClockRate) {
  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  device Dev = Ctx.get_devices()[0];

  auto MemoryClockRate =
      Dev.get_info<ext::intel::info::device::memory_clock_rate>();

  EXPECT_EQ(MemoryClockRate, 800u);
  EXPECT_EQ(sizeof(MemoryClockRate), sizeof(uint32_t))
      << "Expect memory_clock_rate to be of uint32_t size";
}

TEST_F(DeviceInfoTest, GetDeviceMemoryBusWidth) {
  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  device Dev = Ctx.get_devices()[0];

  auto MemoryBusWidth =
      Dev.get_info<ext::intel::info::device::memory_bus_width>();

  EXPECT_EQ(MemoryBusWidth, 64u);
  EXPECT_EQ(sizeof(MemoryBusWidth), sizeof(uint32_t))
      << "Expect memory_bus_width to be of uint32_t size";
}

TEST_F(DeviceInfoTest, BuiltInKernelIDs) {
  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));
  TestContext->BuiltInKernels = "Kernel0;Kernel1;Kernel2";

  device Dev = Ctx.get_devices()[0];

  auto ids = Dev.get_info<info::device::built_in_kernel_ids>();

  ASSERT_EQ(ids.size(), 3u);
  EXPECT_STREQ(ids[0].get_name(), "Kernel0");
  EXPECT_STREQ(ids[1].get_name(), "Kernel1");
  EXPECT_STREQ(ids[2].get_name(), "Kernel2");

  errc val = errc::success;
  std::string msg;
  try {
    get_kernel_bundle<bundle_state::executable>(Ctx, {Dev}, ids);
  } catch (sycl::exception &e) {
    val = errc(e.code().value());
    msg = e.what();
  }

  EXPECT_EQ(val, errc::kernel_argument);
  EXPECT_EQ(
      msg, "Attempting to use a built-in kernel. They are not fully supported");
}

TEST_F(DeviceInfoNegativeTest, TestAspectNotSupported) {
  context Ctx{Plt.get_devices()[0]};
  device Dev = Ctx.get_devices()[0];

  EXPECT_EQ(Dev.has(aspect::ext_intel_device_info_uuid), false);
  EXPECT_EQ(Dev.has(aspect::ext_intel_free_memory), false);
  EXPECT_EQ(Dev.has(aspect::ext_intel_memory_clock_rate), false);
  EXPECT_EQ(Dev.has(aspect::ext_intel_memory_bus_width), false);
}

TEST_F(DeviceInfoTest, SplitStringDelimeterSpace) {
  std::string InputString("V1 V2 V3");
  std::vector<std::string> Expected{"V1", "V2", "V3"};
  EXPECT_EQ(detail::split_string(InputString, ' '), Expected);
}

TEST_F(DeviceInfoTest, SplitStringDelimeterSpaceAtTheEnd) {
  std::string InputString("V1 V2 V3 ");
  std::vector<std::string> Expected{"V1", "V2", "V3"};
  EXPECT_EQ(detail::split_string(InputString, ' '), Expected);
}

TEST_F(DeviceInfoTest, SplitStringDelimeterSemicolon) {
  std::string InputString("V1;V2;V3");
  std::vector<std::string> Expected{"V1", "V2", "V3"};
  EXPECT_EQ(detail::split_string(InputString, ';'), Expected);
}

TEST_F(DeviceInfoTest, SplitStringCheckNoDoubleNullCharacters) {
  std::string InputString("V1;V23");
  std::vector<std::string> Result = detail::split_string(InputString, ';');
  EXPECT_EQ(Result[0].length(), (unsigned)2);
  EXPECT_EQ(Result[1].length(), (unsigned)3);
}
