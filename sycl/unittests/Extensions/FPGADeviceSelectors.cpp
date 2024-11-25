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
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

static constexpr char EMULATION_PLATFORM_NAME[] =
    "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
static constexpr char HARDWARE_PLATFORM_NAME[] =
    "Intel(R) FPGA SDK for OpenCL(TM)";

template <const char PlatformName[]> struct RedefTemplatedWrapper {
  static ur_result_t redefinedPlatformGetInfo(void *pParams) {
    auto params = *static_cast<ur_platform_get_info_params_t *>(pParams);
    switch (*params.ppropName) {
    case UR_PLATFORM_INFO_NAME: {
      size_t PlatformNameLen = strlen(PlatformName) + 1;
      if (*params.ppPropValue) {
        assert(*params.ppropSize == PlatformNameLen);
        std::memcpy(*params.ppPropValue, PlatformName, PlatformNameLen);
      }
      if (*params.ppPropSizeRet)
        **params.ppPropSizeRet = PlatformNameLen;
      return UR_RESULT_SUCCESS;
    }
    case UR_PLATFORM_INFO_BACKEND: {
      constexpr auto MockPlatformBackend = UR_PLATFORM_BACKEND_UNKNOWN;
      if (*params.ppPropValue) {
        std::memcpy(*params.ppPropValue, &MockPlatformBackend,
                    sizeof(MockPlatformBackend));
      }
      if (*params.ppPropSizeRet)
        **params.ppPropSizeRet = sizeof(MockPlatformBackend);
      return UR_RESULT_SUCCESS;
    }
    default:
      return UR_RESULT_SUCCESS;
    }
  }
};

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  constexpr char MockDeviceName[] = "Mock FPGA device";
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_TYPE: {
    if (*params.ppPropValue)
      *static_cast<ur_device_type_t *>(*params.ppPropValue) =
          UR_DEVICE_TYPE_FPGA;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(UR_DEVICE_TYPE_FPGA);
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_NAME: {
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(MockDeviceName));
      std::memcpy(*params.ppPropValue, MockDeviceName, sizeof(MockDeviceName));
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(MockDeviceName);
    return UR_RESULT_SUCCESS;
  }
  // Mock FPGA has no sub-devices
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    assert(*params.ppropSize == sizeof(ur_device_affinity_domain_flags_t));
    if (*params.ppPropValue) {
      *static_cast<ur_device_affinity_domain_flags_t *>(*params.ppPropValue) =
          0;
    }
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_SUCCESS;
  }
}

TEST(FPGADeviceSelectorsTest, FPGASelectorTest) {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
  mock::getCallbacks().set_replace_callback(
      "urPlatformGetInfo",
      &RedefTemplatedWrapper<HARDWARE_PLATFORM_NAME>::redefinedPlatformGetInfo);
  sycl::platform Plt = sycl::platform();
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

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
  mock::getCallbacks().set_replace_callback(
      "urPlatformGetInfo",
      &RedefTemplatedWrapper<
          EMULATION_PLATFORM_NAME>::redefinedPlatformGetInfo);
  sycl::platform Plt = sycl::platform();
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

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
  mock::getCallbacks().set_replace_callback(
      "urPlatformGetInfo",
      &RedefTemplatedWrapper<HARDWARE_PLATFORM_NAME>::redefinedPlatformGetInfo);
  sycl::platform Plt = sycl::platform();
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
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
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
