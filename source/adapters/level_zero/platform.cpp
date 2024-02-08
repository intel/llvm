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

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGet(
    ur_adapter_handle_t *, uint32_t,
    uint32_t NumEntries, ///< [in] the number of platforms to be added to
                         ///< phPlatforms. If phPlatforms is not NULL, then
                         ///< NumEntries should be greater than zero, otherwise
                         ///< ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    ur_platform_handle_t
        *Platforms, ///< [out][optional][range(0, NumEntries)] array of handle
                    ///< of platforms. If NumEntries is less than the number of
                    ///< platforms available, then
                    ///< ::urPlatformGet shall only retrieve that number of
                    ///< platforms.
    uint32_t *NumPlatforms ///< [out][optional] returns the total number of
                           ///< platforms available.
) {
  // Platform handles are cached for reuse. This is to ensure consistent
  // handle pointers across invocations and to improve retrieval performance.
  if (const auto *cached_platforms = Adapter.PlatformCache->get_value();
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
    return Adapter.PlatformCache->get_error();
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetInfo(
    ur_platform_handle_t Platform, ///< [in] handle of the platform
    ur_platform_info_t ParamName,  ///< [in] type of the info to retrieve
    size_t Size,      ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *ParamValue, ///< [out][optional] array of bytes holding the info.
                      ///< If Size is not equal to or greater to the real number
                      ///< of bytes needed to return the info then the
                      ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                      ///< pPlatformInfo is not used.
    size_t *SizeRet   ///< [out][optional] pointer to the actual number of bytes
                      ///< being queried by pPlatformInfo.
) {
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
  default:
    urPrint("urPlatformGetInfo: unrecognized ParamName\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t Driver, ///< [in] handle of the platform
    ur_api_version_t *Version    ///< [out] api version
) {
  std::ignore = Driver;
  *Version = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t Platform,     ///< [in] handle of the platform.
    ur_native_handle_t *NativePlatform ///< [out] a pointer to the native
                                       ///< handle of the platform.
) {
  // Extract the Level Zero driver handle from the given PI platform
  *NativePlatform = reinterpret_cast<ur_native_handle_t>(Platform->ZeDriver);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t
        NativePlatform, ///< [in] the native handle of the platform.
    const ur_platform_native_properties_t
        *Properties, ///< [in][optional] pointer to native platform properties
                     ///< struct.
    ur_platform_handle_t *Platform ///< [out] pointer to the handle of the
                                   ///< platform object created.
) {
  std::ignore = Properties;
  auto ZeDriver = ur_cast<ze_driver_handle_t>(NativePlatform);

  uint32_t NumPlatforms = 0;
  ur_adapter_handle_t AdapterHandle = &Adapter;
  UR_CALL(urPlatformGet(&AdapterHandle, 1, 0, nullptr, &NumPlatforms));

  if (NumPlatforms) {
    std::vector<ur_platform_handle_t> Platforms(NumPlatforms);
    UR_CALL(urPlatformGet(&AdapterHandle, 1, NumPlatforms, Platforms.data(),
                          nullptr));

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

ur_result_t ur_platform_handle_t_::initialize() {
  // Cache driver properties
  ZeStruct<ze_driver_properties_t> ZeDriverProperties;
  ZE2UR_CALL(zeDriverGetProperties, (ZeDriver, &ZeDriverProperties));
  uint32_t DriverVersion = ZeDriverProperties.driverVersion;
  // Intel Level-Zero GPU driver stores version as:
  // | 31 - 24 | 23 - 16 | 15 - 0 |
  // |  Major  |  Minor  | Build  |
  auto VersionMajor = std::to_string((DriverVersion & 0xFF000000) >> 24);
  auto VersionMinor = std::to_string((DriverVersion & 0x00FF0000) >> 16);
  auto VersionBuild = std::to_string(DriverVersion & 0x0000FFFF);
  ZeDriverVersion = VersionMajor + "." + VersionMinor + "." + VersionBuild;

  ZE2UR_CALL(zeDriverGetApiVersion, (ZeDriver, &ZeApiVersion));
  ZeDriverApiVersion = std::to_string(ZE_MAJOR_VERSION(ZeApiVersion)) + "." +
                       std::to_string(ZE_MINOR_VERSION(ZeApiVersion));

  // Cache driver extension properties
  uint32_t Count = 0;
  ZE2UR_CALL(zeDriverGetExtensionProperties, (ZeDriver, &Count, nullptr));

  std::vector<ze_driver_extension_properties_t> ZeExtensions(Count);

  ZE2UR_CALL(zeDriverGetExtensionProperties,
             (ZeDriver, &Count, ZeExtensions.data()));

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
    zeDriverExtensionMap[extension.name] = extension.version;
  }

  // Check if import user ptr into USM feature has been requested.
  // If yes, then set up L0 API pointers if the platform supports it.
  ZeUSMImport.setZeUSMImport(this);

  return UR_RESULT_SUCCESS;
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
  return UR_RESULT_SUCCESS;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-ze-opt-disable' for frontend_option = -O0.
// Return '-ze-opt-level=2' for frontend_option = -O1, -O2 or -O3.
// Return '-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'' for
// frontend_option=-ftarget-compile-fast.
UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetBackendOption(
    ur_platform_handle_t Platform, ///< [in] handle of the platform instance.
    const char *FrontendOption, ///< [in] string containing the frontend option.
    const char *
        *PlatformOption ///< [out] returns the correct platform specific
                        ///< compiler option based on the frontend option.
) {
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
  return UR_RESULT_ERROR_INVALID_VALUE;
}
