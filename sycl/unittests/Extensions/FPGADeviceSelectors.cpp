//==------------------- FPGADeviceSelectors.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/fpga_device_selector.hpp>
#include <sycl/sycl.hpp>

#include <detail/config.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <gtest/gtest.h>

static constexpr char EMULATION_PLATFORM_NAME[] =
    "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
static constexpr char HARDWARE_PLATFORM_NAME[] =
    "Intel(R) FPGA SDK for OpenCL(TM)";

template <const char PlatformName[]> struct RedefTemplatedWrapper {
  static pi_result redefinedPlatformGetInfo(pi_platform platform,
                                            pi_platform_info param_name,
                                            size_t param_value_size,
                                            void *param_value,
                                            size_t *param_value_size_ret) {
    switch (param_name) {
    case PI_PLATFORM_INFO_NAME: {
      size_t PlatformNameLen = strlen(PlatformName) + 1;
      if (param_value) {
        assert(param_value_size == PlatformNameLen);
        std::memcpy(param_value, PlatformName, PlatformNameLen);
      }
      if (param_value_size_ret)
        *param_value_size_ret = PlatformNameLen;
      return PI_SUCCESS;
    }
    default:
      return PI_SUCCESS;
    }
  }
};

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  constexpr char MockDeviceName[] = "Mock FPGA device";
  switch (param_name) {
  case PI_DEVICE_INFO_TYPE: {
    if (param_value)
      *static_cast<_pi_device_type *>(param_value) = PI_DEVICE_TYPE_ACC;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(PI_DEVICE_TYPE_ACC);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_NAME: {
    if (param_value) {
      assert(param_value_size == sizeof(MockDeviceName));
      std::memcpy(param_value, MockDeviceName, sizeof(MockDeviceName));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockDeviceName);
    return PI_SUCCESS;
  }
  // Mock FPGA has no sub-devices
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    if (param_value_size_ret) {
      *param_value_size_ret = 0;
    }
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    assert(param_value_size == sizeof(pi_device_affinity_domain));
    if (param_value) {
      *static_cast<pi_device_affinity_domain *>(param_value) = 0;
    }
    return PI_SUCCESS;
  }
  default:
    return PI_SUCCESS;
  }
}

TEST(FPGADeviceSelectorsTest, FPGASelectorTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  sycl::unittest::PiMock Mock;
  Mock.redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  Mock.redefine<detail::PiApiKind::piPlatformGetInfo>(
      RedefTemplatedWrapper<HARDWARE_PLATFORM_NAME>::redefinedPlatformGetInfo);
  sycl::platform Plt = Mock.getPlatform();
  sycl::context Ctx{Plt.get_devices()};

  sycl::queue FPGAQueue{Ctx, sycl::ext::intel::fpga_selector_v};
  EXPECT_EQ(FPGAQueue.get_device(), Plt.get_devices()[0])
      << "Queue did not contain the expected device";

  try {
    sycl::queue EmuFPGAQueue{Ctx, sycl::ext::intel::fpga_emulator_selector_v};
    FAIL() << "Unexpectedly selected emulator device.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime))
        << "Unexpected exception errc.";
  }
}

TEST(FPGADeviceSelectorsTest, FPGAEmulatorSelectorTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  sycl::unittest::PiMock Mock;
  Mock.redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  Mock.redefine<detail::PiApiKind::piPlatformGetInfo>(
      RedefTemplatedWrapper<EMULATION_PLATFORM_NAME>::redefinedPlatformGetInfo);
  sycl::platform Plt = Mock.getPlatform();
  sycl::context Ctx{Plt.get_devices()};

  sycl::queue EmuFPGAQueue{Ctx, sycl::ext::intel::fpga_emulator_selector_v};
  EXPECT_EQ(EmuFPGAQueue.get_device(), Plt.get_devices()[0])
      << "Queue did not contain the expected device";

  try {
    sycl::queue FPGAQueue{Ctx, sycl::ext::intel::fpga_selector_v};
    FAIL() << "Unexpectedly selected non-emulator device.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime))
        << "Unexpected exception errc.";
  }
}

TEST(FPGADeviceSelectorsTest, FPGASimulatorSelectorTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  constexpr char INTELFPGA_ENV[] = "CL_CONTEXT_MPSIM_DEVICE_INTELFPGA";
  ScopedEnvVar EnvVar(INTELFPGA_ENV, nullptr, []() {});

  sycl::unittest::PiMock Mock;
  Mock.redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  Mock.redefine<detail::PiApiKind::piPlatformGetInfo>(
      RedefTemplatedWrapper<HARDWARE_PLATFORM_NAME>::redefinedPlatformGetInfo);
  sycl::platform Plt = Mock.getPlatform();
  sycl::context Ctx{Plt.get_devices()};

  sycl::queue SimuFPGAQueue{Ctx, sycl::ext::intel::fpga_simulator_selector_v};
  EXPECT_EQ(SimuFPGAQueue.get_device(), Plt.get_devices()[0])
      << "Queue did not contain the expected device";

  const char *ReadEnv = getenv(INTELFPGA_ENV);
  EXPECT_NE(ReadEnv, nullptr) << "Environment was unset after call.";
  EXPECT_EQ(std::string(ReadEnv), "1") << "Environment value was not 1";

  try {
    sycl::queue EmuFPGAQueue{Ctx, sycl::ext::intel::fpga_emulator_selector_v};
    FAIL() << "Unexpectedly selected emulator device.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime))
        << "Unexpected exception errc.";
  }
}

TEST(FPGADeviceSelectorsTest, NegativeFPGASelectorTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  constexpr char INTELFPGA_ENV[] = "CL_CONTEXT_MPSIM_DEVICE_INTELFPGA";
  ScopedEnvVar EnvVar(INTELFPGA_ENV, nullptr, []() {});

  // Do not redefine any APIs. We want it to fail for all.
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::context Ctx{Plt.get_devices()};

  try {
    sycl::queue FPGAQueue{Ctx, sycl::ext::intel::fpga_selector_v};
    FAIL() << "Unexpectedly selected non-emulator device.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime))
        << "Unexpected exception errc.";
  }

  try {
    sycl::queue EmuFPGAQueue{Ctx, sycl::ext::intel::fpga_emulator_selector_v};
    FAIL() << "Unexpectedly selected emulator device.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime))
        << "Unexpected exception errc.";
  }

  try {
    sycl::queue SimuFPGAQueue{Ctx, sycl::ext::intel::fpga_simulator_selector_v};
    FAIL() << "Unexpectedly selected simulator device.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime))
        << "Unexpected exception errc.";
  }
}
