//===---------------------------- Filter.cpp
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/exception.hpp>
#include <sycl/platform.hpp>
#include <sycl/sycl.hpp>

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static const ur_platform_handle_t OpenCLPlatform =
    reinterpret_cast<ur_platform_handle_t>(1u);
static const ur_platform_handle_t LevelZeroPlatform =
    reinterpret_cast<ur_platform_handle_t>(2u);
static const ur_device_handle_t OpenCLCpuDevice =
    reinterpret_cast<ur_device_handle_t>(11u);
static const ur_device_handle_t OpenCLGpuDevice =
    reinterpret_cast<ur_device_handle_t>(12u);
static const ur_device_handle_t LevelZeroGpuDevice =
    reinterpret_cast<ur_device_handle_t>(13u);

struct MockPlatformDesc {
  ur_platform_handle_t Handle;
  ur_backend_t Backend;
  std::string Name;
};

struct MockDeviceDesc {
  ur_device_handle_t Handle;
  ur_platform_handle_t Platform;
  ur_device_type_t Type;
  std::string Name;
};

struct MockDeviceInfo {
  std::string PlatformName;
  std::string DeviceName;
  sycl::info::device_type Type;
  sycl::backend Backend;

  bool operator==(const MockDeviceInfo &Other) const {
    return PlatformName == Other.PlatformName &&
           DeviceName == Other.DeviceName && Type == Other.Type &&
           Backend == Other.Backend;
  }
};

struct SelectorTestCase {
  std::string TestName;
  std::string SelectorString;
  std::vector<std::reference_wrapper<const MockDeviceDesc>> ExpectedDevices;
};

const std::array<MockPlatformDesc, 2> MockPlatforms = {{
    {OpenCLPlatform, UR_BACKEND_OPENCL, "Mock OpenCL Platform"},
    {LevelZeroPlatform, UR_BACKEND_LEVEL_ZERO, "Mock Level Zero Platform"},
}};

const std::array<MockDeviceDesc, 3> MockDevices = {{
    {OpenCLCpuDevice, OpenCLPlatform, UR_DEVICE_TYPE_CPU, "Mock OpenCL CPU"},
    {OpenCLGpuDevice, OpenCLPlatform, UR_DEVICE_TYPE_GPU, "Mock OpenCL GPU"},
    {LevelZeroGpuDevice, LevelZeroPlatform, UR_DEVICE_TYPE_GPU,
     "Mock Level Zero GPU"},
}};

const MockDeviceDesc &OpenCLCpuDeviceDesc = MockDevices[0];
const MockDeviceDesc &OpenCLGpuDeviceDesc = MockDevices[1];
const MockDeviceDesc &LevelZeroGpuDeviceDesc = MockDevices[2];

const MockPlatformDesc *FindPlatform(ur_platform_handle_t Handle) {
  auto It = std::find_if(MockPlatforms.begin(), MockPlatforms.end(),
                         [&](const MockPlatformDesc &Platform) {
                           return Platform.Handle == Handle;
                         });
  return It == MockPlatforms.end() ? nullptr : &*It;
}

const MockDeviceDesc *FindDevice(ur_device_handle_t Handle) {
  auto It = std::find_if(
      MockDevices.begin(), MockDevices.end(),
      [&](const MockDeviceDesc &Device) { return Device.Handle == Handle; });
  return It == MockDevices.end() ? nullptr : &*It;
}

MockDeviceInfo GetExpectedSYCLInfo(const MockDeviceDesc &UrDevice) {
  const MockPlatformDesc *UrPlatform = FindPlatform(UrDevice.Platform);

  sycl::backend ExpectedBackend{};
  switch (UrPlatform->Backend) {
  case UR_BACKEND_OPENCL:
    ExpectedBackend = sycl::backend::opencl;
    break;
  case UR_BACKEND_LEVEL_ZERO:
    ExpectedBackend = sycl::backend::ext_oneapi_level_zero;
    break;
  default:
    throw std::runtime_error("Unknown backend in mock mapping");
  }

  sycl::info::device_type ExpectedType{};
  switch (UrDevice.Type) {
  case UR_DEVICE_TYPE_CPU:
    ExpectedType = sycl::info::device_type::cpu;
    break;
  case UR_DEVICE_TYPE_GPU:
    ExpectedType = sycl::info::device_type::gpu;
    break;
  default:
    throw std::runtime_error("Unknown device type in mock mapping");
  }

  return MockDeviceInfo{UrPlatform->Name, UrDevice.Name, ExpectedType,
                        ExpectedBackend};
}

ur_result_t mock_urPlatformGet(void *pParams) {
  auto P = *static_cast<ur_platform_get_params_t *>(pParams);

  if (P.ppNumPlatforms && *P.ppNumPlatforms) {
    **P.ppNumPlatforms = static_cast<uint32_t>(MockPlatforms.size());
  }

  if (P.pphPlatforms && *P.pphPlatforms && P.pNumEntries &&
      *P.pNumEntries > 0) {
    const uint32_t Count =
        std::min(*P.pNumEntries, static_cast<uint32_t>(MockPlatforms.size()));
    for (uint32_t I = 0; I < Count; ++I) {
      (*P.pphPlatforms)[I] = MockPlatforms[I].Handle;
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urPlatformGetInfo(void *pParams) {
  auto P = *static_cast<ur_platform_get_info_params_t *>(pParams);
  const auto *Plat = FindPlatform(*P.phPlatform);
  if (!Plat) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  if (*P.ppropName == UR_PLATFORM_INFO_BACKEND) {
    if (P.ppPropValue && *P.ppPropValue) {
      *static_cast<ur_backend_t *>(*P.ppPropValue) = Plat->Backend;
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = sizeof(ur_backend_t);
    }
  } else if (*P.ppropName == UR_PLATFORM_INFO_NAME) {
    if (P.ppPropValue && *P.ppPropValue) {
      std::memcpy(*P.ppPropValue, Plat->Name.c_str(), Plat->Name.size() + 1);
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = Plat->Name.size() + 1;
    }
  } else {
    if (P.ppPropValue && *P.ppPropValue) {
      *static_cast<uint32_t *>(*P.ppPropValue) = 0;
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = sizeof(uint32_t);
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urDeviceGet(void *pParams) {
  auto P = *static_cast<ur_device_get_params_t *>(pParams);
  std::vector<ur_device_handle_t> CandidateDevices;

  if (*P.phPlatform == OpenCLPlatform) {
    CandidateDevices = {OpenCLCpuDevice, OpenCLGpuDevice};
  } else if (*P.phPlatform == LevelZeroPlatform) {
    CandidateDevices = {LevelZeroGpuDevice};
  }

  std::vector<ur_device_handle_t> SelectedDevices;
  for (ur_device_handle_t DeviceHandle : CandidateDevices) {
    const MockDeviceDesc *Device = FindDevice(DeviceHandle);
    if (!Device) {
      continue;
    }

    if (*P.pDeviceType != UR_DEVICE_TYPE_ALL &&
        *P.pDeviceType != Device->Type) {
      continue;
    }

    SelectedDevices.push_back(DeviceHandle);
  }

  if (P.ppNumDevices && *P.ppNumDevices) {
    **P.ppNumDevices = static_cast<uint32_t>(SelectedDevices.size());
  }

  if (P.pphDevices && *P.pphDevices && P.pNumEntries && *P.pNumEntries > 0) {
    const uint32_t Count =
        std::min(*P.pNumEntries, static_cast<uint32_t>(SelectedDevices.size()));
    for (uint32_t I = 0; I < Count; ++I) {
      (*P.pphDevices)[I] = SelectedDevices[I];
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urDeviceGetInfo(void *pParams) {
  auto P = *static_cast<ur_device_get_info_params_t *>(pParams);
  const auto *Dev = FindDevice(*P.phDevice);
  if (!Dev) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  switch (*P.ppropName) {
  case UR_DEVICE_INFO_TYPE:
    if (P.ppPropValue && *P.ppPropValue) {
      *static_cast<ur_device_type_t *>(*P.ppPropValue) = Dev->Type;
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = sizeof(ur_device_type_t);
    }
    break;
  case UR_DEVICE_INFO_PLATFORM:
    if (P.ppPropValue && *P.ppPropValue) {
      *static_cast<ur_platform_handle_t *>(*P.ppPropValue) = Dev->Platform;
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = sizeof(ur_platform_handle_t);
    }
    break;
  case UR_DEVICE_INFO_NAME:
    if (P.ppPropValue && *P.ppPropValue) {
      std::memcpy(*P.ppPropValue, Dev->Name.c_str(), Dev->Name.size() + 1);
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = Dev->Name.size() + 1;
    }
    break;
  default:
    if (P.ppPropValue && *P.ppPropValue && P.ppropSize && *P.ppropSize > 0) {
      std::memset(*P.ppPropValue, 0, *P.ppropSize);
    }
    if (P.ppPropSizeRet && *P.ppPropSizeRet) {
      **P.ppPropSizeRet = sizeof(uint32_t);
    }
    break;
  }
  return UR_RESULT_SUCCESS;
}

class OneAPIDeviceSelectorTest : public ::testing::Test {
protected:
  void SetUp() override {
    mock::getCallbacks().set_replace_callback("urPlatformGet",
                                              &mock_urPlatformGet);
    mock::getCallbacks().set_replace_callback("urPlatformGetInfo",
                                              &mock_urPlatformGetInfo);
    mock::getCallbacks().set_replace_callback("urDeviceGet", &mock_urDeviceGet);
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                              &mock_urDeviceGetInfo);
  }

  std::vector<MockDeviceInfo> GetAllDevices() {
    std::vector<MockDeviceInfo> Devices;

    for (const auto &Platform : sycl::platform::get_platforms()) {
      auto PlatformName = Platform.get_info<sycl::info::platform::name>();

      for (const auto &Device : Platform.get_devices()) {
        Devices.push_back(MockDeviceInfo{
            PlatformName, Device.get_info<sycl::info::device::name>(),
            Device.get_info<sycl::info::device::device_type>(),
            Device.get_backend()});
      }
    }

    return Devices;
  }

private:
  sycl::unittest::UrMock<> Mock;
};

class OneAPIDeviceSelectorParamTest
    : public OneAPIDeviceSelectorTest,
      public ::testing::WithParamInterface<SelectorTestCase> {};

TEST_P(OneAPIDeviceSelectorParamTest, CheckFiltering) {
  const auto &[TestName, SelectorString, ExpectedDevices] = GetParam();

  sycl::unittest::ScopedEnvVar SelectorEnv(
      "ONEAPI_DEVICE_SELECTOR", SelectorString.c_str(), []() {
        sycl::detail::SYCLConfig<sycl::detail::ONEAPI_DEVICE_SELECTOR>::reset();
      });

  auto ActualDevices = GetAllDevices();

  std::vector<MockDeviceInfo> ExpectedInfos;
  for (const auto &DeviceDesc : ExpectedDevices) {
    ExpectedInfos.push_back(GetExpectedSYCLInfo(DeviceDesc.get()));
  }

  ASSERT_EQ(ActualDevices.size(), ExpectedInfos.size());

  for (const auto &Expected : ExpectedInfos) {
    auto It = std::find(ActualDevices.begin(), ActualDevices.end(), Expected);
    EXPECT_NE(It, ActualDevices.end())
        << "Failed to find Expected device: " << Expected.DeviceName
        << " on platform: " << Expected.PlatformName;
  }
}

INSTANTIATE_TEST_SUITE_P(
    ValidSelectors, OneAPIDeviceSelectorParamTest,
    ::testing::Values(
        SelectorTestCase{
            "FindAllDevices",
            "*:*",
            {OpenCLCpuDeviceDesc, OpenCLGpuDeviceDesc, LevelZeroGpuDeviceDesc}},
        SelectorTestCase{"FindAllCpuDevices", "*:cpu", {OpenCLCpuDeviceDesc}},
        SelectorTestCase{"FindAllGpuDevices",
                         "*:gpu",
                         {OpenCLGpuDeviceDesc, LevelZeroGpuDeviceDesc}},
        SelectorTestCase{"FindOpenCLDevices",
                         "opencl:*",
                         {OpenCLCpuDeviceDesc, OpenCLGpuDeviceDesc}},
        SelectorTestCase{
            "FindLevelZeroDevices", "level_zero:*", {LevelZeroGpuDeviceDesc}},
        SelectorTestCase{
            "FindLevelZeroGpuOnly", "level_zero:gpu", {LevelZeroGpuDeviceDesc}},

        SelectorTestCase{"ExcludeCpuDevices",
                         "*:*;!*:cpu",
                         {OpenCLGpuDeviceDesc, LevelZeroGpuDeviceDesc}},

        SelectorTestCase{
            "ExcludeGpuDevices", "*:*;!*:gpu", {OpenCLCpuDeviceDesc}},

        SelectorTestCase{
            "ExcludeOpenCLDevices", "*:*;!opencl:*", {LevelZeroGpuDeviceDesc}},
        SelectorTestCase{"ExcludeLevelZeroDevices",
                         "*:*;!level_zero:*",
                         {OpenCLCpuDeviceDesc, OpenCLGpuDeviceDesc}},
        SelectorTestCase{"OpenCLCpuAndLevelZeroGpu",
                         "opencl:cpu;level_zero:gpu",
                         {OpenCLCpuDeviceDesc, LevelZeroGpuDeviceDesc}},
        SelectorTestCase{"MultipleSelectorsForSameDevice",
                         "opencl:cpu;*:cpu",
                         {OpenCLCpuDeviceDesc}},
        SelectorTestCase{"AllGpusButNotLevelZero",
                         "*:gpu;!level_zero:*",
                         {OpenCLGpuDeviceDesc}}

        ),
    [](const ::testing::TestParamInfo<SelectorTestCase> &Info) {
      return Info.param.TestName;
    });

} // namespace
