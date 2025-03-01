//===--------- platform.cpp - Level Zero Adapter --------------------------===//
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
#include "ur_level_zero.hpp"

namespace ur::level_zero {

ur_result_t urPlatformGet(
    ur_adapter_handle_t *, uint32_t,
    /// [in] the number of platforms to be added to phPlatforms. If phPlatforms
    /// is not NULL, then NumEntries should be greater than zero, otherwise
    /// ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)] array of handle of platforms.
    /// If NumEntries is less than the number of platforms available, then
    /// ::urPlatformGet shall only retrieve that number of platforms.
    ur_platform_handle_t *Platforms,
    /// [out][optional] returns the total number of platforms available.
    uint32_t *NumPlatforms) {
  // Platform handles are cached for reuse. This is to ensure consistent
  // handle pointers across invocations and to improve retrieval performance.
  if (const auto *cached_platforms = GlobalAdapter->PlatformCache->get_value();
      cached_platforms) {
    uint32_t nplatforms = (uint32_t)cached_platforms->size();
    if (NumPlatforms) {
      *NumPlatforms = nplatforms;
    }
    if (Platforms) {
      for (uint32_t i = 0; i < std::min(nplatforms, NumEntries); ++i) {
        Platforms[i] = cached_platforms->at(i).get();
      }
    }
  } else {
    return GlobalAdapter->PlatformCache->get_error();
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urPlatformGetInfo(
    /// [in] handle of the platform
    ur_platform_handle_t Platform,
    /// [in] type of the info to retrieve
    ur_platform_info_t ParamName,
    /// [in] the number of bytes pointed to by pPlatformInfo.
    size_t Size,
    /// [out][optional] array of bytes holding the info. If Size is not equal to
    /// or greater to the real number of bytes needed to return the info then
    /// the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and pPlatformInfo
    /// is not used.
    void *ParamValue,
    /// [out][optional] pointer to the actual number of bytes being queried by
    /// pPlatformInfo.
    size_t *SizeRet) {
  UrReturnHelper ReturnValue(Size, ParamValue, SizeRet);

  switch (ParamName) {
  case UR_PLATFORM_INFO_NAME:
    // TODO: Query Level Zero driver when relevant info is added there.
    return ReturnValue("Intel(R) oneAPI Unified Runtime over Level-Zero");
  case UR_PLATFORM_INFO_VENDOR_NAME:
    // TODO: Query Level Zero driver when relevant info is added there.
    return ReturnValue("Intel(R) Corporation");
  case UR_PLATFORM_INFO_EXTENSIONS:
    // Convention adopted from OpenCL:
    //     "Returns a space-separated list of extension names (the extension
    // names themselves do not contain any spaces) supported by the platform.
    // Extensions defined here must be supported by all devices associated
    // with this platform."
    //
    // TODO: Check the common extensions supported by all connected devices and
    // return them. For now, hardcoding some extensions we know are supported by
    // all Level Zero devices.
    return ReturnValue(ZE_SUPPORTED_EXTENSIONS);
  case UR_PLATFORM_INFO_PROFILE:
    // TODO: figure out what this means and how is this used
    return ReturnValue("FULL_PROFILE");
  case UR_PLATFORM_INFO_VERSION:
    // TODO: this should query to zeDriverGetDriverVersion
    // but we don't yet have the driver handle here.
    //
    // From OpenCL 2.1: "This version string has the following format:
    // OpenCL<space><major_version.minor_version><space><platform-specific
    // information>. Follow the same notation here.
    //
    return ReturnValue(Platform->ZeDriverApiVersion.c_str());
  case UR_PLATFORM_INFO_BACKEND:
    return ReturnValue(UR_PLATFORM_BACKEND_LEVEL_ZERO);
  case UR_PLATFORM_INFO_ADAPTER:
    return ReturnValue(GlobalAdapter);
  default:
    logger::debug("urPlatformGetInfo: unrecognized ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urPlatformGetApiVersion(
    /// [in] handle of the platform
    ur_platform_handle_t Driver,
    /// [out] api version
    ur_api_version_t *Version) {
  std::ignore = Driver;
  *Version = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

ur_result_t urPlatformGetNativeHandle(
    /// [in] handle of the platform.
    ur_platform_handle_t Platform,
    /// [out] a pointer to the native handle of the platform.
    ur_native_handle_t *NativePlatform) {
  // Extract the Level Zero driver handle from the given PI platform
  *NativePlatform = reinterpret_cast<ur_native_handle_t>(Platform->ZeDriver);
  return UR_RESULT_SUCCESS;
}

ur_result_t urPlatformCreateWithNativeHandle(
    /// [in] the native handle of the platform.
    ur_native_handle_t NativePlatform, ur_adapter_handle_t,
    /// [in][optional] pointer to native platform properties struct.
    const ur_platform_native_properties_t *Properties,
    /// [out] pointer to the handle of the platform object created.
    ur_platform_handle_t *Platform) {
  std::ignore = Properties;
  auto ZeDriver = ur_cast<ze_driver_handle_t>(NativePlatform);

  uint32_t NumPlatforms = 0;
  ur_adapter_handle_t AdapterHandle = GlobalAdapter;
  UR_CALL(ur::level_zero::urPlatformGet(&AdapterHandle, 1, 0, nullptr,
                                        &NumPlatforms));

  if (NumPlatforms) {
    std::vector<ur_platform_handle_t> Platforms(NumPlatforms);
    UR_CALL(ur::level_zero::urPlatformGet(&AdapterHandle, 1, NumPlatforms,
                                          Platforms.data(), nullptr));

    // The SYCL spec requires that the set of platforms must remain fixed for
    // the duration of the application's execution. We assume that we found all
    // of the Level Zero drivers when we initialized the platform cache, so the
    // "NativeHandle" must already be in the cache. If it is not, this must not
    // be a valid Level Zero driver.
    for (const ur_platform_handle_t &CachedPlatform : Platforms) {
      if (CachedPlatform->ZeDriver == ZeDriver) {
        *Platform = CachedPlatform;
        return UR_RESULT_SUCCESS;
      }
    }
  }

  return UR_RESULT_ERROR_INVALID_VALUE;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-ze-opt-disable' for frontend_option = -O0.
// Return '-ze-opt-level=2' for frontend_option = -O1, -O2 or -O3.
// Return '-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'' for
// frontend_option=-ftarget-compile-fast.
ur_result_t urPlatformGetBackendOption(
    /// [in] handle of the platform instance.
    ur_platform_handle_t Platform,
    /// [in] string containing the frontend option.
    const char *FrontendOption,
    /// [out] returns the correct platform specific compiler option based on
    /// the frontend option.
    const char **PlatformOption) {
  std::ignore = Platform;
  using namespace std::literals;
  if (FrontendOption == nullptr) {
    return UR_RESULT_SUCCESS;
  }
  if (FrontendOption == ""sv) {
    *PlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  if (FrontendOption == "-O0"sv) {
    *PlatformOption = "-ze-opt-disable";
    return UR_RESULT_SUCCESS;
  }
  if (FrontendOption == "-O1"sv || FrontendOption == "-O2"sv ||
      FrontendOption == "-O3"sv) {
    *PlatformOption = "-ze-opt-level=2";
    return UR_RESULT_SUCCESS;
  }
  if (FrontendOption == "-ftarget-compile-fast"sv) {
    *PlatformOption = "-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'";
    return UR_RESULT_SUCCESS;
  }
  if (FrontendOption == "-foffload-fp32-prec-div"sv ||
      FrontendOption == "-foffload-fp32-prec-sqrt"sv) {
    *PlatformOption = "-ze-fp32-correctly-rounded-divide-sqrt";
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

} // namespace ur::level_zero

ur_result_t ur_platform_handle_t_::initialize() {
  ZE2UR_CALL(zeDriverGetApiVersion, (ZeDriver, &ZeApiVersion));
  ZeDriverApiVersion = std::to_string(ZE_MAJOR_VERSION(ZeApiVersion)) + "." +
                       std::to_string(ZE_MINOR_VERSION(ZeApiVersion));

  // Cache driver extension properties
  uint32_t Count = 0;
  ZE2UR_CALL(zeDriverGetExtensionProperties, (ZeDriver, &Count, nullptr));

  std::vector<ze_driver_extension_properties_t> ZeExtensions(Count);

  ZE2UR_CALL(zeDriverGetExtensionProperties,
             (ZeDriver, &Count, ZeExtensions.data()));

  bool MutableCommandListSpecExtensionSupported = false;
  bool ZeIntelExternalSemaphoreExtensionSupported = false;
  bool ZeImmediateCommandListAppendExtensionFound = false;
  for (auto &extension : ZeExtensions) {
    // Check if global offset extension is available
    if (strncmp(extension.name, ZE_GLOBAL_OFFSET_EXP_NAME,
                strlen(ZE_GLOBAL_OFFSET_EXP_NAME) + 1) == 0) {
      if (extension.version == ZE_GLOBAL_OFFSET_EXP_VERSION_1_0) {
        ZeDriverGlobalOffsetExtensionFound = true;
      }
    }
    // Check if extension is available for "static linking" (compiling multiple
    // SPIR-V modules together into one Level Zero module).
    if (strncmp(extension.name, ZE_MODULE_PROGRAM_EXP_NAME,
                strlen(ZE_MODULE_PROGRAM_EXP_NAME) + 1) == 0) {
      if (extension.version == ZE_MODULE_PROGRAM_EXP_VERSION_1_0) {
        ZeDriverModuleProgramExtensionFound = true;
      }
    }
    // Check if extension is available for Counting Events.
    if (strncmp(extension.name, ZE_EVENT_POOL_COUNTER_BASED_EXP_NAME,
                strlen(ZE_EVENT_POOL_COUNTER_BASED_EXP_NAME) + 1) == 0) {
      if (extension.version ==
          ZE_EVENT_POOL_COUNTER_BASED_EXP_VERSION_CURRENT) {
        ZeDriverEventPoolCountingEventsExtensionFound = true;
      }
    }
    // Check if the ImmediateAppendCommandLists extension is available.
    if (strncmp(extension.name, ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_NAME,
                strlen(ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_NAME) + 1) == 0) {
      if (extension.version ==
          ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_VERSION_CURRENT) {
        ZeImmediateCommandListAppendExtensionFound = true;
      }
    }
    // Check if extension is available for Mutable Command List v1.1.
    if (strncmp(extension.name, ZE_MUTABLE_COMMAND_LIST_EXP_NAME,
                strlen(ZE_MUTABLE_COMMAND_LIST_EXP_NAME) + 1) == 0) {
      if (extension.version == ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1) {
        MutableCommandListSpecExtensionSupported = true;
      }
    }
    // Check if extension is available for External Sempahores
    if (strncmp(extension.name, ZE_INTEL_EXTERNAL_SEMAPHORE_EXP_NAME,
                strlen(ZE_INTEL_EXTERNAL_SEMAPHORE_EXP_NAME) + 1) == 0) {
      if (extension.version == ZE_EXTERNAL_SEMAPHORE_EXP_VERSION_1_0) {
        ZeIntelExternalSemaphoreExtensionSupported = true;
      }
    }
    if (strncmp(extension.name, ZE_EU_COUNT_EXT_NAME,
                strlen(ZE_EU_COUNT_EXT_NAME) + 1) == 0) {
      if (extension.version == ZE_EU_COUNT_EXT_VERSION_1_0) {
        ZeDriverEuCountExtensionFound = true;
      }
    }
    if (strncmp(extension.name,
                ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME,
                strlen(ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME) +
                    1) == 0) {
      if (extension.version ==
          ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_1_0) {
        ZeCopyOffloadExtensionSupported = true;
      }
    }
    zeDriverExtensionMap[extension.name] = extension.version;
  }

  ZE2UR_CALL(zelLoaderTranslateHandle, (ZEL_HANDLE_DRIVER, ZeDriver,
                                        (void **)&ZeDriverHandleExpTranslated));

  // Check if intel Driver Version Extension is supported.
  ZeDriverVersionString.setZeDriverVersionString(this);
  // Cache driver properties
  ZeStruct<ze_driver_properties_t> ZeDriverProperties;
  ZE2UR_CALL(zeDriverGetProperties, (ZeDriver, &ZeDriverProperties));
  if (!ZeDriverVersionString.Supported) {
    uint32_t DriverVersion = ZeDriverProperties.driverVersion;
    // Intel Level-Zero GPU driver stores version as:
    // | 31 - 24 | 23 - 16 | 15 - 0 |
    // |  Major  |  Minor  | Build  |
    auto VersionMajor = std::to_string((DriverVersion & 0xFF000000) >> 24);
    auto VersionMinor = std::to_string((DriverVersion & 0x00FF0000) >> 16);
    auto VersionBuild = std::to_string(DriverVersion & 0x0000FFFF);
    ZeDriverVersion = VersionMajor + "." + VersionMinor + "." + VersionBuild;
  } else {
    size_t sizeOfDriverString = 0;
    ZeDriverVersionString.getDriverVersionString(ZeDriverHandleExpTranslated,
                                                 nullptr, &sizeOfDriverString);
    ZeDriverVersion.resize(sizeOfDriverString);
    ZeDriverVersionString.getDriverVersionString(ZeDriverHandleExpTranslated,
                                                 ZeDriverVersion.data(),
                                                 &sizeOfDriverString);
  }

  // Check if import user ptr into USM feature has been requested.
  // If yes, then set up L0 API pointers if the platform supports it.
  ZeUSMImport.setZeUSMImport(this);

  if (ZeIntelExternalSemaphoreExtensionSupported) {
    ZeExternalSemaphoreExt.Supported |=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeIntelDeviceImportExternalSemaphoreExp",
              reinterpret_cast<void **>(
                  &ZeExternalSemaphoreExt.zexImportExternalSemaphoreExp))) ==
         0);
    ZeExternalSemaphoreExt.Supported |=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeIntelCommandListAppendWaitExternalSemaphoresExp",
              reinterpret_cast<void **>(
                  &ZeExternalSemaphoreExt
                       .zexCommandListAppendWaitExternalSemaphoresExp))) == 0);
    ZeExternalSemaphoreExt.Supported |=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeIntelCommandListAppendSignalExternalSemaphoresExp",
              reinterpret_cast<void **>(
                  &ZeExternalSemaphoreExt
                       .zexCommandListAppendSignalExternalSemaphoresExp))) ==
         0);
    ZeExternalSemaphoreExt.Supported |=
        (ZE_CALL_NOCHECK(zeDriverGetExtensionFunctionAddress,
                         (ZeDriver, "zeIntelDeviceReleaseExternalSemaphoreExp",
                          reinterpret_cast<void **>(
                              &ZeExternalSemaphoreExt
                                   .zexDeviceReleaseExternalSemaphoreExp))) ==
         0);
  }

  // Check if mutable command list extension is supported and initialize
  // function pointers.
  if (MutableCommandListSpecExtensionSupported) {
#ifdef UR_STATIC_LEVEL_ZERO
    ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp =
        zeCommandListGetNextCommandIdExp;
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandsExp =
        zeCommandListUpdateMutableCommandsExp;
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandSignalEventExp =
        zeCommandListUpdateMutableCommandSignalEventExp;
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandWaitEventsExp =
        zeCommandListUpdateMutableCommandWaitEventsExp;
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandKernelsExp =
        zeCommandListUpdateMutableCommandKernelsExp;
    ZeMutableCmdListExt.zexCommandListGetNextCommandIdWithKernelsExp =
        zeCommandListGetNextCommandIdWithKernelsExp;
#else
    ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp =
        (ze_pfnCommandListGetNextCommandIdExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListGetNextCommandIdExp");
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandsExp =
        (ze_pfnCommandListUpdateMutableCommandsExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListUpdateMutableCommandsExp");
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandSignalEventExp =
        (ze_pfnCommandListUpdateMutableCommandSignalEventExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListUpdateMutableCommandSignalEventExp");
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandWaitEventsExp =
        (ze_pfnCommandListUpdateMutableCommandWaitEventsExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListUpdateMutableCommandWaitEventsExp");
    ZeMutableCmdListExt.zexCommandListUpdateMutableCommandKernelsExp =
        (ze_pfnCommandListUpdateMutableCommandKernelsExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListUpdateMutableCommandKernelsExp");
    ZeMutableCmdListExt.zexCommandListGetNextCommandIdWithKernelsExp =
        (ze_pfnCommandListGetNextCommandIdWithKernelsExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListGetNextCommandIdWithKernelsExp");
#endif
    ZeMutableCmdListExt.Supported |=
        ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp != nullptr;
    ZeMutableCmdListExt.Supported |=
        ZeMutableCmdListExt.zexCommandListGetNextCommandIdWithKernelsExp !=
        nullptr;
    ZeMutableCmdListExt.Supported |=
        ZeMutableCmdListExt.zexCommandListUpdateMutableCommandKernelsExp !=
        nullptr;
    ZeMutableCmdListExt.Supported |=
        ZeMutableCmdListExt.zexCommandListUpdateMutableCommandWaitEventsExp !=
        nullptr;
    ZeMutableCmdListExt.Supported |=
        ZeMutableCmdListExt.zexCommandListUpdateMutableCommandSignalEventExp !=
        nullptr;
    ZeMutableCmdListExt.Supported |=
        ZeMutableCmdListExt.zexCommandListUpdateMutableCommandsExp != nullptr;
    ZeMutableCmdListExt.LoaderExtension = true;
  } else {
    ZeMutableCmdListExt.Supported |=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeCommandListGetNextCommandIdExp",
              reinterpret_cast<void **>(
                  &ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp))) ==
         0);

    ZeMutableCmdListExt.Supported &=
        (ZE_CALL_NOCHECK(zeDriverGetExtensionFunctionAddress,
                         (ZeDriver, "zeCommandListUpdateMutableCommandsExp",
                          reinterpret_cast<void **>(
                              &ZeMutableCmdListExt
                                   .zexCommandListUpdateMutableCommandsExp))) ==
         0);

    ZeMutableCmdListExt.Supported &=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeCommandListUpdateMutableCommandSignalEventExp",
              reinterpret_cast<void **>(
                  &ZeMutableCmdListExt
                       .zexCommandListUpdateMutableCommandSignalEventExp))) ==
         0);

    ZeMutableCmdListExt.Supported &=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeCommandListUpdateMutableCommandWaitEventsExp",
              reinterpret_cast<void **>(
                  &ZeMutableCmdListExt
                       .zexCommandListUpdateMutableCommandWaitEventsExp))) ==
         0);
    ZeMutableCmdListExt.Supported &=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeCommandListUpdateMutableCommandKernelsExp",
              reinterpret_cast<void **>(
                  &ZeMutableCmdListExt
                       .zexCommandListUpdateMutableCommandKernelsExp))) == 0);

    ZeMutableCmdListExt.Supported &=
        (ZE_CALL_NOCHECK(
             zeDriverGetExtensionFunctionAddress,
             (ZeDriver, "zeCommandListGetNextCommandIdWithKernelsExp",
              reinterpret_cast<void **>(
                  &ZeMutableCmdListExt
                       .zexCommandListGetNextCommandIdWithKernelsExp))) == 0);
  }

  // Check if ImmediateAppendCommandList is supported and initialize the
  // function pointer.
  if (ZeImmediateCommandListAppendExtensionFound) {
#ifdef UR_STATIC_LEVEL_ZERO
    ZeCommandListImmediateAppendExt
        .zeCommandListImmediateAppendCommandListsExp =
        zeCommandListImmediateAppendCommandListsExp;
#else
    ZeCommandListImmediateAppendExt
        .zeCommandListImmediateAppendCommandListsExp =
        (ze_pfnCommandListImmediateAppendCommandListsExp_t)
            ur_loader::LibLoader::getFunctionPtr(
                GlobalAdapter->processHandle,
                "zeCommandListImmediateAppendCommandListsExp");
#endif
    ZeCommandListImmediateAppendExt.Supported =
        ZeCommandListImmediateAppendExt
            .zeCommandListImmediateAppendCommandListsExp != nullptr;
  }

  return UR_RESULT_SUCCESS;
}

/// Checks the version of the level-zero driver.
/// @param VersionMajor Major verion number to compare to.
/// @param VersionMinor Minor verion number to compare to.
/// @param VersionBuild Build verion number to compare to.
/// @return true is the version of the driver is higher than or equal to the
/// compared version
bool ur_platform_handle_t_::isDriverVersionNewerOrSimilar(
    uint32_t VersionMajor, uint32_t VersionMinor, uint32_t VersionBuild) {
  uint32_t DriverVersionMajor = 0;
  uint32_t DriverVersionMinor = 0;
  uint32_t DriverVersionBuild = 0;
  if (!ZeDriverVersionString.Supported) {
    ZeStruct<ze_driver_properties_t> ZeDriverProperties;
    ZE2UR_CALL(zeDriverGetProperties, (ZeDriver, &ZeDriverProperties));
    uint32_t DriverVersion = ZeDriverProperties.driverVersion;
    DriverVersionMajor = (DriverVersion & 0xFF000000) >> 24;
    DriverVersionMinor = (DriverVersion & 0x00FF0000) >> 16;
    DriverVersionBuild = DriverVersion & 0x0000FFFF;
  } else {
    std::string ZeDriverVersion;
    size_t sizeOfDriverString = 0;
    ZeDriverVersionString.getDriverVersionString(ZeDriverHandleExpTranslated,
                                                 nullptr, &sizeOfDriverString);
    ZeDriverVersion.resize(sizeOfDriverString);
    ZeDriverVersionString.getDriverVersionString(ZeDriverHandleExpTranslated,
                                                 ZeDriverVersion.data(),
                                                 &sizeOfDriverString);

    // Intel driver version string is in the format:
    // Major.Minor.Build+Hotfix where hotfix is optional.
    std::stringstream VersionString(ZeDriverVersion);

    std::string VersionValue;
    std::vector<std::string> VersionValues;
    char VersionDelim = '.';
    char HotfixDelim = '+';

    while (getline(VersionString, VersionValue, VersionDelim)) {
      VersionValues.push_back(VersionValue);
    }
    // If the extension exists, but the string value comes by empty or
    // malformed, assume this is a developer driver.
    if (VersionValues.size() >= 3) {
      DriverVersionMajor = atoi(VersionValues[0].c_str());
      DriverVersionMinor = atoi(VersionValues[1].c_str());
      std::stringstream HotfixString(VersionValues[2]);
      std::vector<std::string> BuildHotfixVersionValues;
      // Check to see if there is a hotfix value and strip it off.
      while (getline(HotfixString, VersionValue, HotfixDelim)) {
        BuildHotfixVersionValues.push_back(VersionValue);
      }
      DriverVersionBuild = atoi(BuildHotfixVersionValues[0].c_str());
    } else {
      return true;
    }
  }
  return std::make_tuple(DriverVersionMajor, DriverVersionMinor,
                         DriverVersionBuild) >=
         std::make_tuple(VersionMajor, VersionMinor, VersionBuild);
}

// Get the cached PI device created for the L0 device handle.
// Return NULL if no such PI device found.
ur_device_handle_t
ur_platform_handle_t_::getDeviceFromNativeHandle(ze_device_handle_t ZeDevice) {

  ur_result_t Res = populateDeviceCacheIfNeeded();
  if (Res != UR_RESULT_SUCCESS) {
    return nullptr;
  }

  // TODO: our sub-sub-device representation is currently [Level-Zero device
  // handle + Level-Zero compute group/engine index], so there is now no 1:1
  // mapping from L0 device handle to PI device assumed in this function. Until
  // Level-Zero adds unique ze_device_handle_t for sub-sub-devices, here we
  // filter out PI sub-sub-devices.
  std::shared_lock<ur_shared_mutex> Lock(URDevicesCacheMutex);
  auto it = std::find_if(URDevicesCache.begin(), URDevicesCache.end(),
                         [&](std::unique_ptr<ur_device_handle_t_> &D) {
                           return D.get()->ZeDevice == ZeDevice &&
                                  (D.get()->RootDevice == nullptr ||
                                   D.get()->RootDevice->RootDevice == nullptr);
                         });
  if (it != URDevicesCache.end()) {
    return (*it).get();
  }
  return nullptr;
}

// Check the device cache and load it if necessary.
ur_result_t ur_platform_handle_t_::populateDeviceCacheIfNeeded() {
  std::scoped_lock<ur_shared_mutex> Lock(URDevicesCacheMutex);

  if (DeviceCachePopulated) {
    return UR_RESULT_SUCCESS;
  }

  uint32_t ZeDeviceCount = 0;
  ZE2UR_CALL(zeDeviceGet, (ZeDriver, &ZeDeviceCount, nullptr));

  try {
    std::vector<ze_device_handle_t> ZeDevices(ZeDeviceCount);
    ZE2UR_CALL(zeDeviceGet, (ZeDriver, &ZeDeviceCount, ZeDevices.data()));

    for (uint32_t I = 0; I < ZeDeviceCount; ++I) {
      std::unique_ptr<ur_device_handle_t_> Device(
          new ur_device_handle_t_(ZeDevices[I], (ur_platform_handle_t)this));
      UR_CALL(Device->initialize());

      // Additionally we need to cache all sub-devices too, such that they
      // are readily visible to the urDeviceCreateWithNativeHandle.
      //
      uint32_t SubDevicesCount = 0;
      ZE2UR_CALL(zeDeviceGetSubDevices,
                 (Device->ZeDevice, &SubDevicesCount, nullptr));

      auto ZeSubdevices = new ze_device_handle_t[SubDevicesCount];
      ZE2UR_CALL(zeDeviceGetSubDevices,
                 (Device->ZeDevice, &SubDevicesCount, ZeSubdevices));

      // Wrap the Level Zero sub-devices into PI sub-devices, and add them to
      // cache.
      for (uint32_t I = 0; I < SubDevicesCount; ++I) {
        std::unique_ptr<ur_device_handle_t_> UrSubDevice(
            new ur_device_handle_t_(ZeSubdevices[I], (ur_platform_handle_t)this,
                                    Device.get()));
        auto Result = UrSubDevice->initialize();
        if (Result != UR_RESULT_SUCCESS) {
          delete[] ZeSubdevices;
          return Result;
        }

        // collect all the ordinals for the sub-sub-devices
        std::vector<int> Ordinals;

        uint32_t numQueueGroups = 0;
        ZE2UR_CALL(zeDeviceGetCommandQueueGroupProperties,
                   (UrSubDevice->ZeDevice, &numQueueGroups, nullptr));
        if (numQueueGroups == 0) {
          return UR_RESULT_ERROR_UNKNOWN;
        }
        std::vector<ZeStruct<ze_command_queue_group_properties_t>>
            QueueGroupProperties(numQueueGroups);
        ZE2UR_CALL(zeDeviceGetCommandQueueGroupProperties,
                   (UrSubDevice->ZeDevice, &numQueueGroups,
                    QueueGroupProperties.data()));

        for (uint32_t i = 0; i < numQueueGroups; i++) {
          if (QueueGroupProperties[i].flags &
                  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE &&
              QueueGroupProperties[i].numQueues > 1) {
            Ordinals.push_back(i);
          }
        }

        // If isn't PVC, then submissions to different CCS can be executed on
        // the same EUs still, so we cannot treat them as sub-sub-devices.
        if (UrSubDevice->isPVC() || ExposeCSliceInAffinityPartitioning) {
          // Create PI sub-sub-devices with the sub-device for all the ordinals.
          // Each {ordinal, index} points to a specific CCS which constructs
          // a sub-sub-device at this point.
          //
          // FIXME: Level Zero creates multiple UrDevices for a single physical
          // device when sub-device is partitioned into sub-sub-devices.
          // Sub-sub-device is technically a command queue and we should not
          // build program for each command queue. UrDevice is probably not the
          // right abstraction for a Level Zero command queue.
          for (uint32_t J = 0; J < Ordinals.size(); ++J) {
            for (uint32_t K = 0;
                 K < QueueGroupProperties[Ordinals[J]].numQueues; ++K) {
              std::unique_ptr<ur_device_handle_t_> URSubSubDevice(
                  new ur_device_handle_t_(ZeSubdevices[I],
                                          (ur_platform_handle_t)this,
                                          UrSubDevice.get()));
              UR_CALL(URSubSubDevice->initialize(Ordinals[J], K));

              // save pointers to sub-sub-devices for quick retrieval in the
              // future.
              UrSubDevice->SubDevices.push_back(URSubSubDevice.get());
              URDevicesCache.push_back(std::move(URSubSubDevice));
            }
          }
        }

        // save pointers to sub-devices for quick retrieval in the future.
        Device->SubDevices.push_back(UrSubDevice.get());
        URDevicesCache.push_back(std::move(UrSubDevice));
      }
      delete[] ZeSubdevices;

      // When using ZE_FLAT_DEVICE_HIERARCHY=COMBINED, zeDeviceGet will
      // return tiles as devices, but we can get the card device handle
      // through zeDeviceGetRootDevice. We need to cache the card device
      // handle too, such that it is readily visible to the
      // urDeviceCreateWithNativeHandle.
      ze_device_handle_t RootDevice = nullptr;
      // We cannot use ZE2UR_CALL because under some circumstances this call may
      // return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, and ZE2UR_CALL will abort
      // because it's not UR_RESULT_SUCCESS. Instead, we use ZE_CALL_NOCHECK and
      // we check manually that the result is either ZE_RESULT_SUCCESS or
      // ZE_RESULT_ERROR_UNSUPPORTED_FEATURE.
      auto errc = ZE_CALL_NOCHECK(zeDeviceGetRootDevice,
                                  (Device->ZeDevice, &RootDevice));
      if (errc != ZE_RESULT_SUCCESS &&
          errc != ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
        return ze2urResult(errc);

      if (RootDevice) {
        if (std::find_if(URDevicesCache.begin(), URDevicesCache.end(),
                         [&](auto &Dev) {
                           return Dev->ZeDevice == RootDevice;
                         }) == URDevicesCache.end()) {
          std::unique_ptr<ur_device_handle_t_> UrRootDevice(
              new ur_device_handle_t_(RootDevice, (ur_platform_handle_t)this));
          UR_CALL(UrRootDevice->initialize());
          URDevicesCache.push_back(std::move(UrRootDevice));
        }
      }

      // Save the root device in the cache for future uses.
      URDevicesCache.push_back(std::move(Device));
    }
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  DeviceCachePopulated = true;

  size_t id = 0;
  for (auto &dev : URDevicesCache) {
    dev->Id = id++;
  }

  return UR_RESULT_SUCCESS;
}

size_t ur_platform_handle_t_::getNumDevices() { return URDevicesCache.size(); }

ur_device_handle_t ur_platform_handle_t_::getDeviceById(DeviceId id) {
  for (auto &dev : URDevicesCache) {
    if (dev->Id == id) {
      return dev.get();
    }
  }
  return nullptr;
}
