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
#include <format>
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
    mock_platforms();
    mock_devices();
  }

  void TearDown() override {
    sycl::detail::GlobalHandler::resetGlobalHandler();
  }

  void mock_platforms() {
    mock::getCallbacks().set_replace_callback("urPlatformGet",
                                              &mock_urPlatformGet);
    mock::getCallbacks().set_replace_callback("urPlatformGetInfo",
                                              &mock_urPlatformGetInfo);
  }

  void mock_devices() {
    mock::getCallbacks().set_replace_callback("urDeviceGet", &mock_urDeviceGet);
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                              &mock_urDeviceGetInfo);
  }

private:
  sycl::unittest::UrMock<> Mock;
};

TEST_F(OneAPIDeviceSelectorTest, Test1) {
  sycl::unittest::ScopedEnvVar SelectorEnv(
      "ONEAPI_DEVICE_SELECTOR", "*:cpu", []() {
        sycl::detail::SYCLConfig<sycl::detail::ONEAPI_DEVICE_SELECTOR>{}
            .reset();
      });
  auto platforms = sycl::platform::get_platforms();
  ASSERT_FALSE(platforms.empty());
  for (auto platform : platforms) {
    auto devices = platform.get_devices();
    ASSERT_FALSE(devices.empty());
    for (auto device : devices) {
      auto name = device.get_info<sycl::info::device::name>();
      auto type = device.get_info<sycl::info::device::device_type>();
      std::cout << "Platform: "
                << platform.get_info<sycl::info::platform::name>()
                << ", Device: " << name << ", Type: "
                << (type == sycl::info::device_type::cpu
                        ? "CPU"
                        : (type == sycl::info::device_type::gpu ? "GPU"
                                                                : "Other"))
                << std::endl;
    }
  }
}

TEST_F(OneAPIDeviceSelectorTest, Test2) {
  sycl::unittest::ScopedEnvVar SelectorEnv(
      "ONEAPI_DEVICE_SELECTOR", "*:gpu", []() {
        sycl::detail::SYCLConfig<sycl::detail::ONEAPI_DEVICE_SELECTOR>{}
            .reset();
      });
  auto platforms = sycl::platform::get_platforms();
  ASSERT_FALSE(platforms.empty());
  for (auto platform : platforms) {
    auto devices = platform.get_devices();
    ASSERT_FALSE(devices.empty());
    for (auto device : devices) {
      auto name = device.get_info<sycl::info::device::name>();
      auto type = device.get_info<sycl::info::device::device_type>();
      std::cout << "Platform: "
                << platform.get_info<sycl::info::platform::name>()
                << ", Device: " << name << ", Type: "
                << (type == sycl::info::device_type::cpu
                        ? "CPU"
                        : (type == sycl::info::device_type::gpu ? "GPU"
                                                                : "Other"))
                << std::endl;
    }
  }
}

} // namespace
