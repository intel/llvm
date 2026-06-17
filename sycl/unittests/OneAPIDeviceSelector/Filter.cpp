//===---------------------------- Filter.cpp
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <helpers/UrMock.hpp>
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

const MockPlatformDesc *findPlatform(ur_platform_handle_t Handle) {
  auto It = std::find_if(MockPlatforms.begin(), MockPlatforms.end(),
                         [&](const MockPlatformDesc &Platform) {
                           return Platform.Handle == Handle;
                         });
  return It == MockPlatforms.end() ? nullptr : &*It;
}

const MockDeviceDesc *findDevice(ur_device_handle_t Handle) {
  auto It = std::find_if(
      MockDevices.begin(), MockDevices.end(),
      [&](const MockDeviceDesc &Device) { return Device.Handle == Handle; });
  return It == MockDevices.end() ? nullptr : &*It;
}

MockDeviceInfo getExpectedSYCLInfo(const MockDeviceDesc &UrDevice) {
  const MockPlatformDesc *UrPlatform = findPlatform(UrDevice.Platform);

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
  auto Params = *static_cast<ur_platform_get_params_t *>(pParams);
  if (*Params.ppNumPlatforms)
    **Params.ppNumPlatforms = static_cast<uint32_t>(MockPlatforms.size());

  if (*Params.pphPlatforms && *Params.pNumEntries > 0) {
    const uint32_t Count = std::min<uint32_t>(
        *Params.pNumEntries, static_cast<uint32_t>(MockPlatforms.size()));
    for (uint32_t I = 0; I < Count; ++I)
      (*Params.pphPlatforms)[I] = MockPlatforms[I].Handle;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urPlatformGetInfo(void *pParams) {
  auto Params = *static_cast<ur_platform_get_info_params_t *>(pParams);
  const MockPlatformDesc *Platform = findPlatform(*Params.phPlatform);
  if (!Platform)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;

  switch (*Params.ppropName) {
  case UR_PLATFORM_INFO_BACKEND:
    if (*Params.ppPropValue) {
      auto BackendPtr = static_cast<ur_backend_t *>(*Params.ppPropValue);
      *BackendPtr = Platform->Backend;
    }
    if (*Params.ppPropSizeRet) {

      **Params.ppPropSizeRet = sizeof(ur_backend_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_PLATFORM_INFO_NAME:
    if (*Params.ppPropValue) {
      std::memcpy(*Params.ppPropValue, Platform->Name.c_str(),
                  Platform->Name.size() + 1);
    }
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = Platform->Name.size() + 1;
    return UR_RESULT_SUCCESS;
  default:
    if (*Params.ppPropValue)
      *static_cast<char *>(*Params.ppPropValue) = '\0';
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = 1;
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t mock_urDeviceGet(void *pParams) {
  auto Params = *static_cast<ur_device_get_params_t *>(pParams);
  std::vector<ur_device_handle_t> CandidateDevices;

  if (*Params.phPlatform == OpenCLPlatform) {
    CandidateDevices = {OpenCLCpuDevice, OpenCLGpuDevice};
  } else if (*Params.phPlatform == LevelZeroPlatform) {
    CandidateDevices = {LevelZeroGpuDevice};
  }

  std::vector<ur_device_handle_t> SelectedDevices;
  for (ur_device_handle_t DeviceHandle : CandidateDevices) {
    const MockDeviceDesc *Device = findDevice(DeviceHandle);
    if (!Device)
      continue;

    if (*Params.pDeviceType != UR_DEVICE_TYPE_ALL &&
        *Params.pDeviceType != Device->Type)
      continue;

    SelectedDevices.push_back(DeviceHandle);
  }

  if (*Params.ppNumDevices)
    **Params.ppNumDevices = static_cast<uint32_t>(SelectedDevices.size());

  if (*Params.pphDevices && *Params.pNumEntries > 0) {
    const uint32_t Count = std::min<uint32_t>(
        *Params.pNumEntries, static_cast<uint32_t>(SelectedDevices.size()));
    for (uint32_t I = 0; I < Count; ++I)
      (*Params.pphDevices)[I] = SelectedDevices[I];
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  const MockDeviceDesc *Device = findDevice(*Params.phDevice);
  if (!Device)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;

  switch (*Params.ppropName) {
  case UR_DEVICE_INFO_TYPE:
    if (*Params.ppPropValue)
      *static_cast<ur_device_type_t *>(*Params.ppPropValue) = Device->Type;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_device_type_t);
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_PLATFORM:
    if (*Params.ppPropValue)
      *static_cast<ur_platform_handle_t *>(*Params.ppPropValue) =
          Device->Platform;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_platform_handle_t);
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_NAME:
    if (*Params.ppPropValue) {
      std::memcpy(*Params.ppPropValue, Device->Name.c_str(),
                  Device->Name.size() + 1);
    }
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = Device->Name.size() + 1;
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_PARENT_DEVICE:
    if (*Params.ppPropValue)
      *static_cast<ur_device_handle_t *>(*Params.ppPropValue) = nullptr;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_device_handle_t);
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_AVAILABLE:
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
    if (*Params.ppPropValue)
      *static_cast<ur_bool_t *>(*Params.ppPropValue) = true;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_bool_t);
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_EXTENSIONS:
    if (*Params.ppPropValue)
      *static_cast<char *>(*Params.ppPropValue) = '\0';
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = 1;
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS:
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = 0;
    return UR_RESULT_SUCCESS;
  default:
    if (*Params.ppPropValue && *Params.ppropSize != 0)
      std::memset(*Params.ppPropValue, 0, *Params.ppropSize);
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = 1;
    return UR_RESULT_SUCCESS;
  }
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

  std::vector<MockDeviceInfo> getAllDevices() {
    std::vector<MockDeviceInfo> devices;

    for (const auto &platform : sycl::platform::get_platforms()) {
      auto platformName = platform.get_info<sycl::info::platform::name>();

      for (const auto &device : platform.get_devices()) {
        devices.push_back(MockDeviceInfo{
            platformName, device.get_info<sycl::info::device::name>(),
            device.get_info<sycl::info::device::device_type>(),
            device.get_backend()});
      }
    }

    return devices;
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

  auto actualDevices = getAllDevices();

  std::vector<MockDeviceInfo> expectedInfos;
  for (const auto &DeviceDesc : ExpectedDevices) {
    expectedInfos.push_back(getExpectedSYCLInfo(DeviceDesc.get()));
  }

  ASSERT_EQ(actualDevices.size(), expectedInfos.size());

  for (const auto &expected : expectedInfos) {
    auto it = std::find(actualDevices.begin(), actualDevices.end(), expected);
    EXPECT_NE(it, actualDevices.end())
        << "Failed to find expected device: " << expected.DeviceName
        << " on backend: " << expected.PlatformName;
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
