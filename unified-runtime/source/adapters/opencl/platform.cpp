//===--------- platform.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.hpp"
#include "adapter.hpp"
#include "device.hpp"

static cl_int mapURPlatformInfoToCL(ur_platform_info_t URPropName) {

  switch (URPropName) {
  case UR_PLATFORM_INFO_NAME:
    return CL_PLATFORM_NAME;
  case UR_PLATFORM_INFO_VENDOR_NAME:
    return CL_PLATFORM_VENDOR;
  case UR_PLATFORM_INFO_VERSION:
    return CL_PLATFORM_VERSION;
  case UR_PLATFORM_INFO_EXTENSIONS:
    return CL_PLATFORM_EXTENSIONS;
  case UR_PLATFORM_INFO_PROFILE:
    return CL_PLATFORM_PROFILE;
  default:
    return -1;
  }
}

static bool isBannedOpenCLPlatform(cl_platform_id platform) {
  size_t nameSize = 0;
  cl_int res =
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &nameSize);
  if (res != CL_SUCCESS || nameSize == 0) {
    return false;
  }

  std::string name(nameSize, '\0');
  res = clGetPlatformInfo(platform, CL_PLATFORM_NAME, nameSize, name.data(),
                          nullptr);
  if (res != CL_SUCCESS) {
    return false;
  }

  // The NVIDIA OpenCL platform is currently not compatible with DPC++
  // since it is only 1.2 but gets selected by default in many systems.
  // There is also no support on the PTX backend for OpenCL consumption.
  //
  // There is also no support for the AMD HSA backend for OpenCL consumption,
  // as well as reported problems with device queries, so AMD OpenCL support
  // is disabled as well.
  bool isBanned =
      name.find("NVIDIA CUDA") != std::string::npos ||
      name.find("AMD Accelerated Parallel Processing") != std::string::npos;

  return isBanned;
}

static bool isBannedOpenCLDevice(cl_device_id device) {
  cl_device_type deviceType = 0;
  cl_int res = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type),
                               &deviceType, nullptr);
  if (res != CL_SUCCESS) {
    return false;
  }

  // Filter out FPGA accelerator devices as their usage with OpenCL adapter is
  // deprecated
  bool isBanned = (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0;

  return isBanned;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t hPlatform, ur_platform_info_t propName,
                  size_t propSize, void *pPropValue, size_t *pSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pSizeRet);
  const cl_int CLPropName = mapURPlatformInfoToCL(propName);

  switch (static_cast<uint32_t>(propName)) {
  case UR_PLATFORM_INFO_BACKEND:
    return ReturnValue(UR_BACKEND_OPENCL);
  case UR_PLATFORM_INFO_ADAPTER:
    return ReturnValue(ur::cl::getAdapter());
  case UR_PLATFORM_INFO_NAME:
  case UR_PLATFORM_INFO_VENDOR_NAME:
  case UR_PLATFORM_INFO_VERSION:
  case UR_PLATFORM_INFO_EXTENSIONS:
  case UR_PLATFORM_INFO_PROFILE: {
    cl_platform_id Plat = hPlatform->CLPlatform;

    CL_RETURN_ON_FAILURE(
        clGetPlatformInfo(Plat, CLPropName, propSize, pPropValue, pSizeRet));

    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGetApiVersion([[maybe_unused]] ur_platform_handle_t hPlatform,
                        ur_api_version_t *pVersion) {
  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(ur_adapter_handle_t, uint32_t NumEntries,
              ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {
  static std::mutex adapterPopulationMutex{};
  ur_adapter_handle_t Adapter = nullptr;
  UR_RETURN_ON_FAILURE(urAdapterGet(1, &Adapter, nullptr));
  if (!Adapter) {
    // The only operation urAdapterGet really performs is allocating the adapter
    // handle via new, so no adapter handle here almost certainly means memory
    // problems.
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  if (Adapter->NumPlatforms == 0) {
    std::lock_guard guard{adapterPopulationMutex};

    // It's possible for urPlatformGet, if ran on multiple threads, to enter
    // this branch simultaneously. This check ensures that only one sees that
    // Adapter->NumPlatforms is zero.
    if (Adapter->NumPlatforms == 0) {
      uint32_t NumPlatforms = 0;
      cl_int Res = clGetPlatformIDs(0, nullptr, &NumPlatforms);

      if (NumPlatforms == 0 || Res == CL_PLATFORM_NOT_FOUND_KHR) {
        if (pNumPlatforms) {
          *pNumPlatforms = 0;
        }
        return UR_RESULT_SUCCESS;
      }
      CL_RETURN_ON_FAILURE(Res);

      std::vector<cl_platform_id> CLPlatforms(NumPlatforms);
      Res = clGetPlatformIDs(static_cast<cl_uint>(NumPlatforms),
                             CLPlatforms.data(), nullptr);
      CL_RETURN_ON_FAILURE(Res);

      // Filter out banned platforms
      std::vector<cl_platform_id> FilteredPlatforms;
      for (uint32_t i = 0; i < NumPlatforms; i++) {
        if (!isBannedOpenCLPlatform(CLPlatforms[i])) {
          FilteredPlatforms.push_back(CLPlatforms[i]);
        }
      }

      try {
        for (auto &Platform : FilteredPlatforms) {
          auto URPlatform = std::make_unique<ur_platform_handle_t_>(Platform);
          UR_RETURN_ON_FAILURE(URPlatform->InitDevices());
          // Only add platforms that have devices, especially in case all
          // devices are banned
          if (!URPlatform->Devices.empty()) {
            Adapter->URPlatforms.emplace_back(URPlatform.release());
          }
        }
        Adapter->NumPlatforms =
            static_cast<uint32_t>(Adapter->URPlatforms.size());
      } catch (std::bad_alloc &) {
        return UR_RESULT_ERROR_OUT_OF_RESOURCES;
      } catch (...) {
        return UR_RESULT_ERROR_INVALID_PLATFORM;
      }
    }
  }

  if (pNumPlatforms != nullptr) {
    *pNumPlatforms = Adapter->NumPlatforms;
  }
  if (NumEntries && phPlatforms) {
    for (uint32_t i = 0; i < NumEntries; i++) {
      phPlatforms[i] = Adapter->URPlatforms[i].get();
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ur_native_handle_t *phNativePlatform) {
  *phNativePlatform =
      reinterpret_cast<ur_native_handle_t>(hPlatform->CLPlatform);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ur_adapter_handle_t,
    const ur_platform_native_properties_t *, ur_platform_handle_t *phPlatform) {
  cl_platform_id NativeHandle =
      reinterpret_cast<cl_platform_id>(hNativePlatform);

  uint32_t NumPlatforms = 0;
  UR_RETURN_ON_FAILURE(urPlatformGet(nullptr, 0, nullptr, &NumPlatforms));
  std::vector<ur_platform_handle_t> Platforms(NumPlatforms);
  UR_RETURN_ON_FAILURE(
      urPlatformGet(nullptr, NumPlatforms, Platforms.data(), nullptr));

  for (uint32_t i = 0; i < NumPlatforms; i++) {
    if (Platforms[i]->CLPlatform == NativeHandle) {
      *phPlatform = Platforms[i];
      return UR_RESULT_SUCCESS;
    }
  }
  return UR_RESULT_ERROR_INVALID_PLATFORM;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-cl-opt-disable' for pFrontendOption = -O0 and '' for others.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetBackendOption(ur_platform_handle_t, const char *pFrontendOption,
                           const char **ppPlatformOption) {
  using namespace std::literals;
  if (pFrontendOption == nullptr)
    return UR_RESULT_SUCCESS;
  if (pFrontendOption == ""sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  // Return '-cl-opt-disable' for frontend_option = -O0 and '' for others.
  if (!strcmp(pFrontendOption, "-O0")) {
    *ppPlatformOption = "-cl-opt-disable";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-O1"sv || pFrontendOption == "-O2"sv ||
      pFrontendOption == "-O3"sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-ftarget-compile-fast"sv) {
    *ppPlatformOption = "-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-foffload-fp32-prec-div"sv ||
      pFrontendOption == "-foffload-fp32-prec-sqrt"sv) {
    *ppPlatformOption = "-cl-fp32-correctly-rounded-divide-sqrt";
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

ur_result_t ur_platform_handle_t_::InitDevices() {
  if (Devices.empty()) {
    cl_uint DeviceNum = 0;
    cl_int Res =
        clGetDeviceIDs(CLPlatform, CL_DEVICE_TYPE_ALL, 0, nullptr, &DeviceNum);

    // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
    if (Res == CL_DEVICE_NOT_FOUND) {
      return UR_RESULT_SUCCESS;
    }

    CL_RETURN_ON_FAILURE(Res);

    std::vector<cl_device_id> CLDevices(DeviceNum);
    Res = clGetDeviceIDs(CLPlatform, CL_DEVICE_TYPE_ALL, DeviceNum,
                         CLDevices.data(), nullptr);

    // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
    if (Res == CL_DEVICE_NOT_FOUND) {
      return UR_RESULT_SUCCESS;
    }

    CL_RETURN_ON_FAILURE(Res);

    // Filter out banned devices
    std::vector<cl_device_id> FilteredDevices;
    for (uint32_t i = 0; i < DeviceNum; i++) {
      if (!isBannedOpenCLDevice(CLDevices[i])) {
        FilteredDevices.push_back(CLDevices[i]);
      }
    }

    try {
      Devices.resize(FilteredDevices.size());
      for (size_t i = 0; i < FilteredDevices.size(); i++) {
        Devices[i] = std::make_unique<ur_device_handle_t_>(FilteredDevices[i],
                                                           this, nullptr);
      }
    } catch (std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }

  return UR_RESULT_SUCCESS;
}
