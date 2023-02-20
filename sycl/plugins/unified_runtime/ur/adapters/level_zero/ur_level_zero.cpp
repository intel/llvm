//===--------- ur_level_zero.hpp - Level Zero Adapter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <string.h>

#include "ur_level_zero.hpp"
#include <ur_bindings.hpp>

// Define the static class field
std::mutex ZeCall::GlobalLock;

ZeUSMImportExtension ZeUSMImport;

void zePrint(const char *Format, ...) {
  if (ZeDebug & ZE_DEBUG_BASIC) {
    va_list Args;
    va_start(Args, Format);
    vfprintf(stderr, Format, Args);
    va_end(Args);
  }
}

// This function will ensure compatibility with both Linux and Windows for
// setting environment variables.
bool setEnvVar(const char *name, const char *value) {
#ifdef _WIN32
  int Res = _putenv_s(name, value);
#else
  int Res = setenv(name, value, 1);
#endif
  if (Res != 0) {
    zePrint(
        "Level Zero plugin was unable to set the environment variable: %s\n",
        name);
    return false;
  }
  return true;
}

// Trace a call to Level-Zero RT
#define ZE_CALL(ZeName, ZeArgs)                                                \
  {                                                                            \
    ze_result_t ZeResult = ZeName ZeArgs;                                      \
    if (auto Result = ZeCall().doCall(ZeResult, #ZeName, #ZeArgs, true))       \
      return ze2urResult(Result);                                              \
  }

// This will count the calls to Level-Zero
std::map<const char *, int> *ZeCallCount = nullptr;

inline void zeParseError(ze_result_t ZeError, const char *&ErrorString) {
  switch (ZeError) {
#define ZE_ERRCASE(ERR)                                                        \
  case ERR:                                                                    \
    ErrorString = "" #ERR;                                                     \
    break;

    ZE_ERRCASE(ZE_RESULT_SUCCESS)
    ZE_ERRCASE(ZE_RESULT_NOT_READY)
    ZE_ERRCASE(ZE_RESULT_ERROR_DEVICE_LOST)
    ZE_ERRCASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY)
    ZE_ERRCASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
    ZE_ERRCASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS)
    ZE_ERRCASE(ZE_RESULT_ERROR_NOT_AVAILABLE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNINITIALIZED)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_ARGUMENT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE)
    ZE_ERRCASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_ENUMERATION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE)
    ZE_ERRCASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNKNOWN)

#undef ZE_ERRCASE
  default:
    assert(false && "Unexpected Error code");
  } // switch
}

ze_result_t ZeCall::doCall(ze_result_t ZeResult, const char *ZeName,
                           const char *ZeArgs, bool TraceError) {
  zePrint("ZE ---> %s%s\n", ZeName, ZeArgs);

  if (ZeDebug & ZE_DEBUG_CALL_COUNT) {
    ++(*ZeCallCount)[ZeName];
  }

  if (ZeResult && TraceError) {
    const char *ErrorString = "Unknown";
    zeParseError(ZeResult, ErrorString);
    zePrint("Error (%s) in %s\n", ErrorString, ZeName);
  }
  return ZeResult;
}

// Specializations for various L0 structures
template <> ze_structure_type_t getZeStructureType<ze_event_pool_desc_t>() {
  return ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_fence_desc_t>() {
  return ZE_STRUCTURE_TYPE_FENCE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_command_list_desc_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_context_desc_t>() {
  return ZE_STRUCTURE_TYPE_CONTEXT_DESC;
}
template <>
ze_structure_type_t
getZeStructureType<ze_relaxed_allocation_limits_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_host_mem_alloc_desc_t>() {
  return ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_mem_alloc_desc_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_command_queue_desc_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_image_desc_t>() {
  return ZE_STRUCTURE_TYPE_IMAGE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_module_desc_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_module_program_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_kernel_desc_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_event_desc_t>() {
  return ZE_STRUCTURE_TYPE_EVENT_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_sampler_desc_t>() {
  return ZE_STRUCTURE_TYPE_SAMPLER_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_driver_properties_t>() {
  return ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_device_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_compute_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_command_queue_group_properties_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_image_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_module_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_cache_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_ext_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_access_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_module_properties_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_kernel_properties_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_memory_allocation_properties_t>() {
  return ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
}

template <> zes_structure_type_t getZesStructureType<zes_pci_properties_t>() {
  return ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
}

template <> zes_structure_type_t getZesStructureType<zes_mem_state_t>() {
  return ZES_STRUCTURE_TYPE_MEM_STATE;
}

template <> zes_structure_type_t getZesStructureType<zes_mem_properties_t>() {
  return ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
}

static const bool ExposeCSliceInAffinityPartitioning = [] {
  const char *Flag =
      std::getenv("SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING");
  return Flag ? std::atoi(Flag) != 0 : false;
}();

zer_result_t _ur_platform_handle_t::initialize() {
  // Cache driver properties
  ZeStruct<ze_driver_properties_t> ZeDriverProperties;
  ZE_CALL(zeDriverGetProperties, (ZeDriver, &ZeDriverProperties));
  uint32_t DriverVersion = ZeDriverProperties.driverVersion;
  // Intel Level-Zero GPU driver stores version as:
  // | 31 - 24 | 23 - 16 | 15 - 0 |
  // |  Major  |  Minor  | Build  |
  auto VersionMajor = std::to_string((DriverVersion & 0xFF000000) >> 24);
  auto VersionMinor = std::to_string((DriverVersion & 0x00FF0000) >> 16);
  auto VersionBuild = std::to_string(DriverVersion & 0x0000FFFF);
  ZeDriverVersion = VersionMajor + "." + VersionMinor + "." + VersionBuild;

  ZE_CALL(zeDriverGetApiVersion, (ZeDriver, &ZeApiVersion));
  ZeDriverApiVersion = std::to_string(ZE_MAJOR_VERSION(ZeApiVersion)) + "." +
                       std::to_string(ZE_MINOR_VERSION(ZeApiVersion));

  // Cache driver extension properties
  uint32_t Count = 0;
  ZE_CALL(zeDriverGetExtensionProperties, (ZeDriver, &Count, nullptr));

  std::vector<ze_driver_extension_properties_t> ZeExtensions(Count);

  ZE_CALL(zeDriverGetExtensionProperties,
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

  return ZER_RESULT_SUCCESS;
}

ZER_APIEXPORT zer_result_t ZER_APICALL zerPlatformGet(
    uint32_t
        *NumPlatforms, ///< [in,out] pointer to the number of platforms.
                       ///< if count is zero, then the call shall update the
                       ///< value with the total number of platforms available.
                       ///< if count is greater than the number of platforms
                       ///< available, then the call shall update the value with
                       ///< the correct number of platforms available.
    zer_platform_handle_t
        *Platforms ///< [out][optional][range(0, *pCount)] array of handle of
                   ///< platforms. if count is less than the number of platforms
                   ///< available, then platform shall only retrieve that number
                   ///< of platforms.
) {
  PI_ASSERT(NumPlatforms, ZER_RESULT_INVALID_VALUE);

  static std::once_flag ZeCallCountInitialized;
  try {
    std::call_once(ZeCallCountInitialized, []() {
      if (ZeDebug & ZE_DEBUG_CALL_COUNT) {
        ZeCallCount = new std::map<const char *, int>;
      }
    });
  } catch (const std::bad_alloc &) {
    return ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return ZER_RESULT_ERROR_UNKNOWN;
  }

  // Setting these environment variables before running zeInit will enable the
  // validation layer in the Level Zero loader.
  if (ZeDebug & ZE_DEBUG_VALIDATION) {
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
    setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
  }

  // Enable SYSMAN support for obtaining the PCI address
  // and maximum memory bandwidth.
  if (getenv("SYCL_ENABLE_PCI") != nullptr) {
    setEnvVar("ZES_ENABLE_SYSMAN", "1");
  }

  // TODO: We can still safely recover if something goes wrong during the init.
  // Implement handling segfault using sigaction.

  // We must only initialize the driver once, even if piPlatformsGet() is called
  // multiple times.  Declaring the return value as "static" ensures it's only
  // called once.
  static ze_result_t ZeResult = ZE_CALL_NOCHECK(zeInit, (0));

  // Absorb the ZE_RESULT_ERROR_UNINITIALIZED and just return 0 Platforms.
  if (ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
    PI_ASSERT(NumPlatforms != 0, ZER_RESULT_INVALID_VALUE);
    *NumPlatforms = 0;
    return ZER_RESULT_SUCCESS;
  }

  if (ZeResult != ZE_RESULT_SUCCESS) {
    zePrint("zeInit: Level Zero initialization failure\n");
    return ze2urResult(ZeResult);
  }

  // Cache pi_platforms for reuse in the future
  // It solves two problems;
  // 1. sycl::platform equality issue; we always return the same pi_platform.
  // 2. performance; we can save time by immediately return from cache.
  //

  const std::lock_guard<SpinLock> Lock{*PiPlatformsCacheMutex};
  if (!PiPlatformCachePopulated) {
    try {
      // Level Zero does not have concept of Platforms, but Level Zero driver is
      // the closest match.
      uint32_t ZeDriverCount = 0;
      ZE_CALL(zeDriverGet, (&ZeDriverCount, nullptr));
      if (ZeDriverCount == 0) {
        PiPlatformCachePopulated = true;
      } else {
        std::vector<ze_driver_handle_t> ZeDrivers;
        ZeDrivers.resize(ZeDriverCount);

        ZE_CALL(zeDriverGet, (&ZeDriverCount, ZeDrivers.data()));
        for (uint32_t I = 0; I < ZeDriverCount; ++I) {
          auto Platform = new _zer_platform_handle_t(ZeDrivers[I]);
          // Save a copy in the cache for future uses.
          PiPlatformsCache->push_back(Platform);

          zer_result_t Result = Platform->initialize();
          if (Result != ZER_RESULT_SUCCESS) {
            return Result;
          }
        }
        PiPlatformCachePopulated = true;
      }
    } catch (const std::bad_alloc &) {
      return ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return ZER_RESULT_ERROR_UNKNOWN;
    }
  }

  // Populate returned platforms from the cache.
  if (Platforms) {
    PI_ASSERT(*NumPlatforms <= PiPlatformsCache->size(),
              ZER_RESULT_INVALID_PLATFORM);
    std::copy_n(PiPlatformsCache->begin(), *NumPlatforms, Platforms);
  }

  if (*NumPlatforms == 0)
    *NumPlatforms = PiPlatformsCache->size();
  else
    *NumPlatforms = std::min(PiPlatformsCache->size(), (size_t)*NumPlatforms);

  return ZER_RESULT_SUCCESS;
}

ZER_APIEXPORT zer_result_t ZER_APICALL zerPlatformGetInfo(
    zer_platform_handle_t Platform, ///< [in] handle of the platform
    zer_platform_info_t ParamName,  ///< [in] type of the info to retrieve
    size_t *pSize, ///< [in,out] pointer to the number of bytes needed to return
                   ///< info queried. the call shall update it with the real
                   ///< number of bytes needed to return the info
    void *ParamValue ///< [out][optional] array of bytes holding the info.
                     ///< if *pSize is not equal to the real number of bytes
                     ///< needed to return the info then the
                     ///< ::ZER_RESULT_ERROR_INVALID_SIZE error is returned and
                     ///< pPlatformInfo is not used.
) {

  PI_ASSERT(Platform, ZER_RESULT_INVALID_PLATFORM);
  UrReturnHelper ReturnValue(pSize, ParamValue);

  switch (ParamName) {
  case ZER_PLATFORM_INFO_NAME:
    // TODO: Query Level Zero driver when relevant info is added there.
    return ReturnValue("Intel(R) oneAPI Unified Runtime over Level-Zero");
  case ZER_PLATFORM_INFO_VENDOR_NAME:
    // TODO: Query Level Zero driver when relevant info is added there.
    return ReturnValue("Intel(R) Corporation");
  case ZER_PLATFORM_INFO_EXTENSIONS:
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
  case ZER_PLATFORM_INFO_PROFILE:
    // TODO: figure out what this means and how is this used
    return ReturnValue("FULL_PROFILE");
  case ZER_PLATFORM_INFO_VERSION:
    // TODO: this should query to zeDriverGetDriverVersion
    // but we don't yet have the driver handle here.
    //
    // From OpenCL 2.1: "This version string has the following format:
    // OpenCL<space><major_version.minor_version><space><platform-specific
    // information>. Follow the same notation here.
    //
    return ReturnValue(Platform->ZeDriverApiVersion.c_str());
  default:
    zePrint("piPlatformGetInfo: unrecognized ParamName\n");
    return ZER_RESULT_INVALID_VALUE;
  }

  return ZER_RESULT_SUCCESS;
}

ZER_APIEXPORT zer_result_t ZER_APICALL zerDeviceGet(
    zer_platform_handle_t Platform, ///< [in] handle of the platform instance
    zer_device_type_t DeviceType,   ///< [in] the type of the devices.
    uint32_t *NumDevices, ///< [in,out] pointer to the number of devices.
                          ///< If count is zero, then the call shall update the
                          ///< value with the total number of devices available.
                          ///< If count is greater than the number of devices
                          ///< available, then the call shall update the value
                          ///< with the correct number of devices available.
    zer_device_handle_t
        *Devices ///< [out][optional][range(0, *pCount)] array of handle of
                 ///< devices. If count is less than the number of devices
                 ///< available, then platform shall only retrieve that number
                 ///< of devices.
) {

  PI_ASSERT(Platform, ZER_RESULT_INVALID_PLATFORM);

  auto Res = Platform->populateDeviceCacheIfNeeded();
  if (Res != ZER_RESULT_SUCCESS) {
    return Res;
  }

  // Filter available devices based on input DeviceType.
  std::vector<zer_device_handle_t> MatchedDevices;
  std::shared_lock<pi_shared_mutex> Lock(Platform->PiDevicesCacheMutex);
  for (auto &D : Platform->PiDevicesCache) {
    // Only ever return root-devices from piDevicesGet, but the
    // devices cache also keeps sub-devices.
    if (D->isSubDevice())
      continue;

    bool Matched = false;
    switch (DeviceType) {
    case ZER_DEVICE_TYPE_ALL:
      Matched = true;
      break;
    case ZER_DEVICE_TYPE_GPU:
    case ZER_DEVICE_TYPE_DEFAULT:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_GPU);
      break;
    case ZER_DEVICE_TYPE_CPU:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_CPU);
      break;
    case ZER_DEVICE_TYPE_FPGA:
      Matched = D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_FPGA;
      break;
    case ZER_DEVICE_TYPE_MCA:
      Matched = D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_MCA;
      break;
    default:
      Matched = false;
      zePrint("Unknown device type");
      break;
    }
    if (Matched)
      MatchedDevices.push_back(D.get());
  }

  uint32_t ZeDeviceCount = MatchedDevices.size();

  if (*NumDevices == 0)
    *NumDevices = ZeDeviceCount;
  else {
    auto N = std::min(ZeDeviceCount, *NumDevices);
    std::copy_n(MatchedDevices.begin(), N, Devices);
  }
  return ZER_RESULT_SUCCESS;
}

ZER_APIEXPORT zer_result_t ZER_APICALL zerDeviceGetInfo(
    zer_device_handle_t Device,  ///< [in] handle of the device instance
    zer_device_info_t ParamName, ///< [in] type of the info to retrieve
    size_t *pSize, ///< [in,out] pointer to the number of bytes needed to return
                   ///< info queried. The call shall update it with the real
                   ///< number of bytes needed to return the info
    void *ParamValue ///< [out][optional] array of bytes holding the info.
                     ///< If *pSize input is not 0 and not equal to the real
                     ///< number of bytes needed to return the info then the
                     ///< ::ZER_RESULT_ERROR_INVALID_SIZE error is returned and
                     ///< pDeviceInfo is not used.
) {
  PI_ASSERT(Device, ZER_RESULT_INVALID_DEVICE);
  UrReturnHelper ReturnValue(pSize, ParamValue);

  ze_device_handle_t ZeDevice = Device->ZeDevice;

  switch ((int)ParamName) {
  case ZER_DEVICE_INFO_TYPE: {
    switch (Device->ZeDeviceProperties->type) {
    case ZE_DEVICE_TYPE_GPU:
      return ReturnValue(ZER_DEVICE_TYPE_GPU);
    case ZE_DEVICE_TYPE_CPU:
      return ReturnValue(ZER_DEVICE_TYPE_CPU);
    case ZE_DEVICE_TYPE_FPGA:
      return ReturnValue(ZER_DEVICE_TYPE_FPGA);
    default:
      zePrint("This device type is not supported\n");
      return ZER_RESULT_INVALID_VALUE;
    }
  }
  case ZER_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(Device->RootDevice);
  case ZER_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case ZER_DEVICE_INFO_VENDOR_ID:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->vendorId});
  case ZER_DEVICE_INFO_UUID: {
    // Intel extension for device UUID. This returns the UUID as
    // std::array<std::byte, 16>. For details about this extension,
    // see sycl/doc/extensions/supported/sycl_ext_intel_device_info.md.
    const auto &UUID = Device->ZeDeviceProperties->uuid.id;
    return ReturnValue(UUID, sizeof(UUID));
  }
  case ZER_DEVICE_INFO_ATOMIC_64:
    return ReturnValue(uint32_t{Device->ZeDeviceModuleProperties->flags &
                                ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS});
  case ZER_DEVICE_INFO_EXTENSIONS: {
    // Convention adopted from OpenCL:
    //     "Returns a space separated list of extension names (the extension
    // names themselves do not contain any spaces) supported by the device."
    //
    // TODO: Use proper mechanism to get this information from Level Zero after
    // it is added to Level Zero.
    // Hardcoding the few we know are supported by the current hardware.
    //
    //
    std::string SupportedExtensions;

    // cl_khr_il_program - OpenCL 2.0 KHR extension for SPIR-V support. Core
    //   feature in >OpenCL 2.1
    // cl_khr_subgroups - Extension adds support for implementation-controlled
    //   subgroups.
    // cl_intel_subgroups - Extension adds subgroup features, defined by Intel.
    // cl_intel_subgroups_short - Extension adds subgroup functions described in
    //   the cl_intel_subgroups extension to support 16-bit integer data types
    //   for performance.
    // cl_intel_required_subgroup_size - Extension to allow programmers to
    //   optionally specify the required subgroup size for a kernel function.
    // cl_khr_fp16 - Optional half floating-point support.
    // cl_khr_fp64 - Support for double floating-point precision.
    // cl_khr_int64_base_atomics, cl_khr_int64_extended_atomics - Optional
    //   extensions that implement atomic operations on 64-bit signed and
    //   unsigned integers to locations in __global and __local memory.
    // cl_khr_3d_image_writes - Extension to enable writes to 3D image memory
    //   objects.
    //
    // Hardcoding some extensions we know are supported by all Level Zero
    // devices.
    SupportedExtensions += (ZE_SUPPORTED_EXTENSIONS);
    if (Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP16)
      SupportedExtensions += ("cl_khr_fp16 ");
    if (Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP64)
      SupportedExtensions += ("cl_khr_fp64 ");
    if (Device->ZeDeviceModuleProperties->flags &
        ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS)
      // int64AtomicsSupported indicates support for both.
      SupportedExtensions +=
          ("cl_khr_int64_base_atomics cl_khr_int64_extended_atomics ");
    if (Device->ZeDeviceImageProperties->maxImageDims3D > 0)
      // Supports reading and writing of images.
      SupportedExtensions += ("cl_khr_3d_image_writes ");

    // L0 does not tell us if bfloat16 is supported.
    // For now, assume ATS and PVC support it.
    // TODO: change the way we detect bfloat16 support.
    if ((Device->ZeDeviceProperties->deviceId & 0xfff) == 0x201 ||
        (Device->ZeDeviceProperties->deviceId & 0xff0) == 0xbd0)
      SupportedExtensions += ("cl_intel_bfloat16_conversions ");

    return ReturnValue(SupportedExtensions.c_str());
  }
  case ZER_DEVICE_INFO_NAME:
    return ReturnValue(Device->ZeDeviceProperties->name);
  // zeModuleCreate allows using root device module for sub-devices:
  // > The application must only use the module for the device, or its
  // > sub-devices, which was provided during creation.
  case ZER_EXT_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return ReturnValue(uint32_t{0});
  case ZER_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(uint32_t{1});
  case ZER_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(uint32_t{1});
  case ZER_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    uint32_t MaxComputeUnits =
        Device->ZeDeviceProperties->numEUsPerSubslice *
        Device->ZeDeviceProperties->numSubslicesPerSlice *
        Device->ZeDeviceProperties->numSlices;

    bool RepresentsCSlice =
        Device->QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute]
            .ZeIndex >= 0;
    if (RepresentsCSlice)
      MaxComputeUnits /= Device->RootDevice->SubDevices.size();

    return ReturnValue(uint32_t{MaxComputeUnits});
  }
  case ZER_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    // Level Zero spec defines only three dimensions
    return ReturnValue(uint32_t{3});
  case ZER_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(
        uint64_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
  case ZER_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{Device->ZeDeviceComputeProperties->maxGroupSizeX,
                       Device->ZeDeviceComputeProperties->maxGroupSizeY,
                       Device->ZeDeviceComputeProperties->maxGroupSizeZ}};
    return ReturnValue(MaxGroupSize);
  }
  case ZER_EXT_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t Arr[3];
    } MaxGroupCounts = {{Device->ZeDeviceComputeProperties->maxGroupCountX,
                         Device->ZeDeviceComputeProperties->maxGroupCountY,
                         Device->ZeDeviceComputeProperties->maxGroupCountZ}};
    return ReturnValue(MaxGroupCounts);
  }
  case ZER_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->coreClockRate});
  case ZER_DEVICE_INFO_ADDRESS_BITS: {
    // TODO: To confirm with spec.
    return ReturnValue(uint32_t{64});
  }
  case ZER_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    return ReturnValue(uint64_t{Device->ZeDeviceProperties->maxMemAllocSize});
  case ZER_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    uint64_t GlobalMemSize = 0;
    for (const auto &ZeDeviceMemoryExtProperty :
         Device->ZeDeviceMemoryProperties->second) {
      GlobalMemSize += ZeDeviceMemoryExtProperty.physicalSize;
    }
    return ReturnValue(uint64_t{GlobalMemSize});
  }
  case ZER_DEVICE_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(
        uint64_t{Device->ZeDeviceComputeProperties->maxSharedLocalMemory});
  case ZER_DEVICE_INFO_IMAGE_SUPPORTED:
    return ReturnValue(
        uint32_t{Device->ZeDeviceImageProperties->maxImageDims1D > 0});
  case ZER_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(uint32_t{(Device->ZeDeviceProperties->flags &
                                 ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0});
  case ZER_DEVICE_INFO_AVAILABLE:
    return ReturnValue(uint32_t{ZeDevice ? true : false});
  case ZER_DEVICE_INFO_VENDOR:
    // TODO: Level-Zero does not return vendor's name at the moment
    // only the ID.
    return ReturnValue("Intel(R) Corporation");
  case ZER_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue(Device->Platform->ZeDriverVersion.c_str());
  case ZER_DEVICE_INFO_VERSION:
    return ReturnValue(Device->Platform->ZeDriverApiVersion.c_str());
  case ZER_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    auto Res = Device->Platform->populateDeviceCacheIfNeeded();
    if (Res != ZER_RESULT_SUCCESS) {
      return Res;
    }
    return ReturnValue((uint32_t)Device->SubDevices.size());
  }
  case ZER_DEVICE_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Device->RefCount.load()});
  case ZER_DEVICE_INFO_PARTITION_PROPERTIES: {
    // SYCL spec says: if this SYCL device cannot be partitioned into at least
    // two sub devices then the returned vector must be empty.
    auto Res = Device->Platform->populateDeviceCacheIfNeeded();
    if (Res != ZER_RESULT_SUCCESS) {
      return Res;
    }

    uint32_t ZeSubDeviceCount = Device->SubDevices.size();
    if (ZeSubDeviceCount < 2) {
      return ReturnValue((zer_device_partition_property_flag_t)0);
    }
    bool PartitionedByCSlice = Device->SubDevices[0]->isCCS();

    auto ReturnHelper = [&](auto... Partitions) {
      struct {
        zer_device_partition_property_flag_t Arr[sizeof...(Partitions) + 1];
      } PartitionProperties = {
          {Partitions..., zer_device_partition_property_flag_t(0)}};
      return ReturnValue(PartitionProperties);
    };

    if (ExposeCSliceInAffinityPartitioning) {
      if (PartitionedByCSlice)
        return ReturnHelper(
            ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE,
            ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN);

      else
        return ReturnHelper(
            ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN);
    } else {
      return ReturnHelper(
          PartitionedByCSlice
              ? ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE
              : ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN);
    }
    break;
  }
  case ZER_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(zer_device_affinity_domain_flag_t(
        ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA |
        ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE));
  case ZER_DEVICE_INFO_PARTITION_TYPE: {
    // For root-device there is no partitioning to report.
    if (!Device->isSubDevice())
      return ReturnValue(zer_device_partition_property_flag_t(0));

    if (Device->isCCS()) {
      struct {
        zer_device_partition_property_flag_t Arr[2];
      } PartitionProperties = {
          {ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE,
           zer_device_partition_property_flag_t(0)}};
      return ReturnValue(PartitionProperties);
    }

    struct {
      zer_device_partition_property_flag_t Arr[3];
    } PartitionProperties = {
        {ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN,
         (zer_device_partition_property_flag_t)
             ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
         zer_device_partition_property_flag_t(0)}};
    return ReturnValue(PartitionProperties);
  }

    // Everything under here is not supported yet

  case ZER_DEVICE_INFO_OPENCL_C_VERSION:
    return ReturnValue("");
  case ZER_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    return ReturnValue(uint32_t{true});
  case ZER_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceModuleProperties->printfBufferSize});
  case ZER_DEVICE_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case ZER_DEVICE_INFO_BUILT_IN_KERNELS:
    // TODO: To find out correct value
    return ReturnValue("");
  case ZER_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(
        zer_queue_flag_t(ZER_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                         ZER_QUEUE_FLAG_PROFILING_ENABLE));
  case ZER_DEVICE_INFO_EXECUTION_CAPABILITIES:
    return ReturnValue(zer_device_exec_capability_flag_t{
        ZER_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL});
  case ZER_DEVICE_INFO_ENDIAN_LITTLE:
    return ReturnValue(uint32_t{true});
  case ZER_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->flags &
                                ZE_DEVICE_PROPERTY_FLAG_ECC});
  case ZER_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    return ReturnValue(size_t{Device->ZeDeviceProperties->timerResolution});
  case ZER_DEVICE_INFO_LOCAL_MEM_TYPE:
    return ReturnValue(ZER_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case ZER_DEVICE_INFO_MAX_CONSTANT_ARGS:
    return ReturnValue(uint32_t{64});
  case ZER_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    return ReturnValue(
        uint64_t{Device->ZeDeviceImageProperties->maxImageBufferSize});
  case ZER_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(ZER_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case ZER_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    return ReturnValue(
        // TODO[1.0]: how to query cache line-size?
        uint32_t{1});
  case ZER_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    return ReturnValue(uint64_t{Device->ZeDeviceCacheProperties->cacheSize});
  case ZER_DEVICE_INFO_MAX_PARAMETER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceModuleProperties->maxArgumentsSize});
  case ZER_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // SYCL/OpenCL spec is vague on what this means exactly, but seems to
    // be for "alignment requirement (in bits) for sub-buffer offsets."
    // An OpenCL implementation returns 8*128, but Level Zero can do just 8,
    // meaning unaligned access for values of types larger than 8 bits.
    return ReturnValue(uint32_t{8});
  case ZER_DEVICE_INFO_MAX_SAMPLERS:
    return ReturnValue(uint32_t{Device->ZeDeviceImageProperties->maxSamplers});
  case ZER_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
    return ReturnValue(
        uint32_t{Device->ZeDeviceImageProperties->maxReadImageArgs});
  case ZER_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    return ReturnValue(
        uint32_t{Device->ZeDeviceImageProperties->maxWriteImageArgs});
  case ZER_DEVICE_INFO_SINGLE_FP_CONFIG: {
    uint64_t SingleFPValue = 0;
    ze_device_fp_flags_t ZeSingleFPCapabilities =
        Device->ZeDeviceModuleProperties->fp32flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeSingleFPCapabilities) {
      SingleFPValue |= ZER_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(uint64_t{SingleFPValue});
  }
  case ZER_DEVICE_INFO_HALF_FP_CONFIG: {
    uint64_t HalfFPValue = 0;
    ze_device_fp_flags_t ZeHalfFPCapabilities =
        Device->ZeDeviceModuleProperties->fp16flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeHalfFPCapabilities) {
      HalfFPValue |= ZER_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(uint64_t{HalfFPValue});
  }
  case ZER_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    uint64_t DoubleFPValue = 0;
    ze_device_fp_flags_t ZeDoubleFPCapabilities =
        Device->ZeDeviceModuleProperties->fp64flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeDoubleFPCapabilities) {
      DoubleFPValue |= ZER_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(uint64_t{DoubleFPValue});
  }
  case ZER_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims2D});
  case ZER_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims2D});
  case ZER_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims3D});
  case ZER_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims3D});
  case ZER_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims3D});
  case ZER_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceImageProperties->maxImageBufferSize});
  case ZER_EXT_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceImageProperties->maxImageArraySlices});
  // Handle SIMD widths.
  // TODO: can we do better than this?
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 1);
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 2);
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 4);
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 8);
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 4);
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 8);
  case ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
  case ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 2);
  case ZER_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Max_num_sub_Groups = maxTotalGroupSize/min(set of subGroupSizes);
    uint32_t MinSubGroupSize =
        Device->ZeDeviceComputeProperties->subGroupSizes[0];
    for (uint32_t I = 1;
         I < Device->ZeDeviceComputeProperties->numSubGroupSizes; I++) {
      if (MinSubGroupSize > Device->ZeDeviceComputeProperties->subGroupSizes[I])
        MinSubGroupSize = Device->ZeDeviceComputeProperties->subGroupSizes[I];
    }
    return ReturnValue(Device->ZeDeviceComputeProperties->maxTotalGroupSize /
                       MinSubGroupSize);
  }
  case ZER_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // TODO: Not supported yet. Needs to be updated after support is added.
    return ReturnValue(uint32_t{false});
  }
  case ZER_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    // ze_device_compute_properties.subGroupSizes is in uint32_t whereas the
    // expected return is size_t datatype. size_t can be 8 bytes of data.
    return ReturnValue.template operator()<size_t>(
        Device->ZeDeviceComputeProperties->subGroupSizes,
        Device->ZeDeviceComputeProperties->numSubGroupSizes);
  }
  case ZER_DEVICE_INFO_IL_VERSION: {
    // Set to a space separated list of IL version strings of the form
    // <IL_Prefix>_<Major_version>.<Minor_version>.
    // "SPIR-V" is a required IL prefix when cl_khr_il_progam extension is
    // reported.
    uint32_t SpirvVersion =
        Device->ZeDeviceModuleProperties->spirvVersionSupported;
    uint32_t SpirvVersionMajor = ZE_MAJOR_VERSION(SpirvVersion);
    uint32_t SpirvVersionMinor = ZE_MINOR_VERSION(SpirvVersion);

    char SpirvVersionString[50];
    int Len = sprintf(SpirvVersionString, "SPIR-V_%d.%d ", SpirvVersionMajor,
                      SpirvVersionMinor);
    // returned string to contain only len number of characters.
    std::string ILVersion(SpirvVersionString, Len);
    return ReturnValue(ILVersion.c_str());
  }
  case ZER_DEVICE_INFO_USM_HOST_SUPPORT:
  case ZER_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case ZER_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case ZER_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case ZER_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    auto MapCaps = [](const ze_memory_access_cap_flags_t &ZeCapabilities) {
      uint64_t Capabilities = 0;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_RW)
        Capabilities |= ZER_EXT_USM_CAPS_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC)
        Capabilities |= ZER_EXT_USM_CAPS_ATOMIC_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT)
        Capabilities |= ZER_EXT_USM_CAPS_CONCURRENT_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC)
        Capabilities |= ZER_EXT_USM_CAPS_CONCURRENT_ATOMIC_ACCESS;
      return Capabilities;
    };
    auto &Props = Device->ZeDeviceMemoryAccessProperties;
    switch (ParamName) {
    case ZER_DEVICE_INFO_USM_HOST_SUPPORT:
      return ReturnValue(MapCaps(Props->hostAllocCapabilities));
    case ZER_DEVICE_INFO_USM_DEVICE_SUPPORT:
      return ReturnValue(MapCaps(Props->deviceAllocCapabilities));
    case ZER_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedSingleDeviceAllocCapabilities));
    case ZER_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedCrossDeviceAllocCapabilities));
    case ZER_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedSystemAllocCapabilities));
    default:
      die("piDeviceGetInfo: enexpected ParamName.");
    }
  }

    // intel extensions for GPU information
  case ZER_EXT_DEVICE_INFO_DEVICE_ID:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->deviceId});
  case ZER_DEVICE_INFO_PCI_ADDRESS: {
    if (getenv("ZES_ENABLE_SYSMAN") == nullptr) {
      zePrint("Set SYCL_ENABLE_PCI=1 to obtain PCI data.\n");
      return ZER_RESULT_INVALID_VALUE;
    }
    ZesStruct<zes_pci_properties_t> ZeDevicePciProperties;
    ZE_CALL(zesDevicePciGetProperties, (ZeDevice, &ZeDevicePciProperties));
    constexpr size_t AddressBufferSize = 13;
    char AddressBuffer[AddressBufferSize];
    std::snprintf(AddressBuffer, AddressBufferSize, "%04x:%02x:%02x.%01x",
                  ZeDevicePciProperties.address.domain,
                  ZeDevicePciProperties.address.bus,
                  ZeDevicePciProperties.address.device,
                  ZeDevicePciProperties.address.function);
    return ReturnValue(AddressBuffer);
  }

  case ZER_EXT_DEVICE_INFO_FREE_MEMORY: {
    if (getenv("ZES_ENABLE_SYSMAN") == nullptr) {
      setErrorMessage("Set ZES_ENABLE_SYSMAN=1 to obtain free memory",
                      ZER_RESULT_SUCCESS);
      return ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR;
    }
    // Only report device memory which zeMemAllocDevice can allocate from.
    // Currently this is only the one enumerated with ordinal 0.
    uint64_t FreeMemory = 0;
    uint32_t MemCount = 0;
    ZE_CALL(zesDeviceEnumMemoryModules, (ZeDevice, &MemCount, nullptr));
    if (MemCount != 0) {
      std::vector<zes_mem_handle_t> ZesMemHandles(MemCount);
      ZE_CALL(zesDeviceEnumMemoryModules,
              (ZeDevice, &MemCount, ZesMemHandles.data()));
      for (auto &ZesMemHandle : ZesMemHandles) {
        ZesStruct<zes_mem_properties_t> ZesMemProperties;
        ZE_CALL(zesMemoryGetProperties, (ZesMemHandle, &ZesMemProperties));
        // For root-device report memory from all memory modules since that
        // is what totally available in the default implicit scaling mode.
        // For sub-devices only report memory local to them.
        if (!Device->isSubDevice() || Device->ZeDeviceProperties->subdeviceId ==
                                          ZesMemProperties.subdeviceId) {

          ZesStruct<zes_mem_state_t> ZesMemState;
          ZE_CALL(zesMemoryGetState, (ZesMemHandle, &ZesMemState));
          FreeMemory += ZesMemState.free;
        }
      }
    }
    return ReturnValue(FreeMemory);
  }
  case ZER_EXT_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    // If there are not any memory modules then return 0.
    if (Device->ZeDeviceMemoryProperties->first.empty())
      return ReturnValue(uint32_t{0});

    // If there are multiple memory modules on the device then we have to report
    // the value of the slowest memory.
    auto Comp = [](const ze_device_memory_properties_t &A,
                   const ze_device_memory_properties_t &B) -> bool {
      return A.maxClockRate < B.maxClockRate;
    };
    auto MinIt =
        std::min_element(Device->ZeDeviceMemoryProperties->first.begin(),
                         Device->ZeDeviceMemoryProperties->first.end(), Comp);
    return ReturnValue(uint32_t{MinIt->maxClockRate});
  }
  case ZER_EXT_DEVICE_INFO_MEMORY_BUS_WIDTH: {
    // If there are not any memory modules then return 0.
    if (Device->ZeDeviceMemoryProperties->first.empty())
      return ReturnValue(uint32_t{0});

    // If there are multiple memory modules on the device then we have to report
    // the value of the slowest memory.
    auto Comp = [](const ze_device_memory_properties_t &A,
                   const ze_device_memory_properties_t &B) -> bool {
      return A.maxBusWidth < B.maxBusWidth;
    };
    auto MinIt =
        std::min_element(Device->ZeDeviceMemoryProperties->first.begin(),
                         Device->ZeDeviceMemoryProperties->first.end(), Comp);
    return ReturnValue(uint32_t{MinIt->maxBusWidth});
  }
  case ZER_EXT_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    if (Device->QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute]
            .ZeIndex >= 0)
      // Sub-sub-device represents a particular compute index already.
      return ReturnValue(int32_t{1});

    auto ZeDeviceNumIndices =
        Device->QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute]
            .ZeProperties.numQueues;
    return ReturnValue(int32_t(ZeDeviceNumIndices));
  } break;
  case ZER_DEVICE_INFO_GPU_EU_COUNT: {
    uint32_t count = Device->ZeDeviceProperties->numEUsPerSubslice *
                     Device->ZeDeviceProperties->numSubslicesPerSlice *
                     Device->ZeDeviceProperties->numSlices;
    return ReturnValue(uint32_t{count});
  }
  case ZER_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
    return ReturnValue(
        uint32_t{Device->ZeDeviceProperties->physicalEUSimdWidth});
  case ZER_EXT_DEVICE_INFO_GPU_SLICES:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->numSlices});
  case ZER_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
    return ReturnValue(
        uint32_t{Device->ZeDeviceProperties->numSubslicesPerSlice});
  case ZER_EXT_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->numEUsPerSubslice});
  case ZER_EXT_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->numThreadsPerEU});
  case ZER_EXT_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    // currently not supported in level zero runtime
    return ZER_RESULT_INVALID_VALUE;
  case ZER_EXT_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS: {
    // bfloat16 math functions are not yet supported on Intel GPUs.
    return ReturnValue(bool{false});
  }

  // TODO: Implement.
  case ZER_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  default:
    zePrint("Unsupported ParamName in piGetDeviceInfo\n");
    zePrint("ParamName=%d(0x%x)\n", ParamName, ParamName);
    return ZER_RESULT_INVALID_VALUE;
  }

  return ZER_RESULT_SUCCESS;
}

// SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE can be set to an integer value, or
// a pair of integer values of the form "lower_index:upper_index".
// Here, the indices point to copy engines in a list of all available copy
// engines.
// This functions returns this pair of indices.
// If the user specifies only a single integer, a value of 0 indicates that
// the copy engines will not be used at all. A value of 1 indicates that all
// available copy engines can be used.
const std::pair<int, int>
getRangeOfAllowedCopyEngines(const zer_device_handle_t &Device) {
  static const char *EnvVar = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE");
  // If the environment variable is not set, no copy engines are used when
  // immediate commandlists are being used. For standard commandlists all are
  // used.
  if (!EnvVar) {
    if (Device->useImmediateCommandLists())
      return std::pair<int, int>(-1, -1);   // No copy engines can be used.
    return std::pair<int, int>(0, INT_MAX); // All copy engines will be used.
  }
  std::string CopyEngineRange = EnvVar;
  // Environment variable can be a single integer or a pair of integers
  // separated by ":"
  auto pos = CopyEngineRange.find(":");
  if (pos == std::string::npos) {
    bool UseCopyEngine = (std::stoi(CopyEngineRange) != 0);
    if (UseCopyEngine)
      return std::pair<int, int>(0, INT_MAX); // All copy engines can be used.
    return std::pair<int, int>(-1, -1);       // No copy engines will be used.
  }
  int LowerCopyEngineIndex = std::stoi(CopyEngineRange.substr(0, pos));
  int UpperCopyEngineIndex = std::stoi(CopyEngineRange.substr(pos + 1));
  if ((LowerCopyEngineIndex > UpperCopyEngineIndex) ||
      (LowerCopyEngineIndex < -1) || (UpperCopyEngineIndex < -1)) {
    zePrint("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE: invalid value provided, "
            "default set.\n");
    LowerCopyEngineIndex = 0;
    UpperCopyEngineIndex = INT_MAX;
  }
  return std::pair<int, int>(LowerCopyEngineIndex, UpperCopyEngineIndex);
}

bool CopyEngineRequested(const zer_device_handle_t &Device) {
  int LowerCopyQueueIndex = getRangeOfAllowedCopyEngines(Device).first;
  int UpperCopyQueueIndex = getRangeOfAllowedCopyEngines(Device).second;
  return ((LowerCopyQueueIndex != -1) || (UpperCopyQueueIndex != -1));
}

// Whether immediate commandlists will be used for kernel launches and copies.
// The default is standard commandlists. Setting 1 or 2 specifies use of
// immediate commandlists. Note: when immediate commandlists are used then
// device-only events must be either AllHostVisible or OnDemandHostVisibleProxy.
// (See env var SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS).

// Get value of immediate commandlists env var setting or -1 if unset
_ur_device_handle_t::ImmCmdlistMode
_ur_device_handle_t::useImmediateCommandLists() {
  // If immediate commandlist setting is not explicitly set, then use the device
  // default.
  static const int ImmediateCommandlistsSetting = [] {
    const char *ImmediateCommandlistsSettingStr =
        std::getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
    if (!ImmediateCommandlistsSettingStr)
      return -1;
    return std::stoi(ImmediateCommandlistsSettingStr);
  }();

  if (ImmediateCommandlistsSetting == -1)
    return ImmCommandListsPreferred ? PerQueue : NotUsed;
  switch (ImmediateCommandlistsSetting) {
  case 0:
    return NotUsed;
  case 1:
    return PerQueue;
  case 2:
    return PerThreadPerQueue;
  default:
    return NotUsed;
  }
}

zer_result_t _ur_device_handle_t::initialize(int SubSubDeviceOrdinal,
                                             int SubSubDeviceIndex) {
  uint32_t numQueueGroups = 0;
  ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
          (ZeDevice, &numQueueGroups, nullptr));
  if (numQueueGroups == 0) {
    return ZER_RESULT_ERROR_UNKNOWN;
  }
  zePrint("NOTE: Number of queue groups = %d\n", numQueueGroups);
  std::vector<ZeStruct<ze_command_queue_group_properties_t>>
      QueueGroupProperties(numQueueGroups);
  ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
          (ZeDevice, &numQueueGroups, QueueGroupProperties.data()));

  // Initialize ordinal and compute queue group properties
  for (uint32_t i = 0; i < numQueueGroups; i++) {
    if (QueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute].ZeOrdinal =
          i;
      QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute]
          .ZeProperties = QueueGroupProperties[i];
      break;
    }
  }

  // Reinitialize a sub-sub-device with its own ordinal, index.
  // Our sub-sub-device representation is currently [Level-Zero sub-device
  // handle + Level-Zero compute group/engine index]. Only the specified
  // index queue will be used to submit work to the sub-sub-device.
  if (SubSubDeviceOrdinal >= 0) {
    QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute].ZeOrdinal =
        SubSubDeviceOrdinal;
    QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute].ZeIndex =
        SubSubDeviceIndex;
  } else { // Proceed with initialization for root and sub-device
    // How is it possible that there are no "compute" capabilities?
    if (QueueGroup[_zer_device_handle_t::queue_group_info_t::Compute]
            .ZeOrdinal < 0) {
      return ZER_RESULT_ERROR_UNKNOWN;
    }

    if (CopyEngineRequested((zer_device_handle_t)this)) {
      for (uint32_t i = 0; i < numQueueGroups; i++) {
        if (((QueueGroupProperties[i].flags &
              ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0) &&
            (QueueGroupProperties[i].flags &
             ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
          if (QueueGroupProperties[i].numQueues == 1) {
            QueueGroup[queue_group_info_t::MainCopy].ZeOrdinal = i;
            QueueGroup[queue_group_info_t::MainCopy].ZeProperties =
                QueueGroupProperties[i];
          } else {
            QueueGroup[queue_group_info_t::LinkCopy].ZeOrdinal = i;
            QueueGroup[queue_group_info_t::LinkCopy].ZeProperties =
                QueueGroupProperties[i];
            break;
          }
        }
      }
      if (QueueGroup[queue_group_info_t::MainCopy].ZeOrdinal < 0)
        zePrint("NOTE: main blitter/copy engine is not available\n");
      else
        zePrint("NOTE: main blitter/copy engine is available\n");

      if (QueueGroup[queue_group_info_t::LinkCopy].ZeOrdinal < 0)
        zePrint("NOTE: link blitter/copy engines are not available\n");
      else
        zePrint("NOTE: link blitter/copy engines are available\n");
    }
  }

  // Maintain various device properties cache.
  // Note that we just describe here how to compute the data.
  // The real initialization is upon first access.
  //
  auto ZeDevice = this->ZeDevice;
  ZeDeviceProperties.Compute = [ZeDevice](ze_device_properties_t &Properties) {
    ZE_CALL_NOCHECK(zeDeviceGetProperties, (ZeDevice, &Properties));
  };

  ZeDeviceComputeProperties.Compute =
      [ZeDevice](ze_device_compute_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetComputeProperties, (ZeDevice, &Properties));
      };

  ZeDeviceImageProperties.Compute =
      [ZeDevice](ze_device_image_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetImageProperties, (ZeDevice, &Properties));
      };

  ZeDeviceModuleProperties.Compute =
      [ZeDevice](ze_device_module_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetModuleProperties, (ZeDevice, &Properties));
      };

  ZeDeviceMemoryProperties.Compute =
      [ZeDevice](
          std::pair<std::vector<ZeStruct<ze_device_memory_properties_t>>,
                    std::vector<ZeStruct<ze_device_memory_ext_properties_t>>>
              &Properties) {
        uint32_t Count = 0;
        ZE_CALL_NOCHECK(zeDeviceGetMemoryProperties,
                        (ZeDevice, &Count, nullptr));

        auto &PropertiesVector = Properties.first;
        auto &PropertiesExtVector = Properties.second;

        PropertiesVector.resize(Count);
        PropertiesExtVector.resize(Count);
        // Request for extended memory properties be read in
        for (uint32_t I = 0; I < Count; ++I)
          PropertiesVector[I].pNext = (void *)&PropertiesExtVector[I];

        ZE_CALL_NOCHECK(zeDeviceGetMemoryProperties,
                        (ZeDevice, &Count, PropertiesVector.data()));
      };

  ZeDeviceMemoryAccessProperties.Compute =
      [ZeDevice](ze_device_memory_access_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetMemoryAccessProperties,
                        (ZeDevice, &Properties));
      };

  ZeDeviceCacheProperties.Compute =
      [ZeDevice](ze_device_cache_properties_t &Properties) {
        // TODO: Since v1.0 there can be multiple cache properties.
        // For now remember the first one, if any.
        uint32_t Count = 0;
        ZE_CALL_NOCHECK(zeDeviceGetCacheProperties,
                        (ZeDevice, &Count, nullptr));
        if (Count > 0)
          Count = 1;
        ZE_CALL_NOCHECK(zeDeviceGetCacheProperties,
                        (ZeDevice, &Count, &Properties));
      };

  // Check device id for PVC.
  // TODO: change mechanism for detecting PVC once L0 provides an interface.
  // At present even PVC doesn't automatically use immediate commandlists.
  // Change this after more testing.
  ImmCommandListsPreferred =
      false; // (ZeDeviceProperties->deviceId & 0xff0) == 0xbd0;

  return ZER_RESULT_SUCCESS;
}

// Get the cached PI device created for the L0 device handle.
// Return NULL if no such PI device found.
zer_device_handle_t
_ur_platform_handle_t::getDeviceFromNativeHandle(ze_device_handle_t ZeDevice) {

  zer_result_t Res = populateDeviceCacheIfNeeded();
  if (Res != ZER_RESULT_SUCCESS) {
    return nullptr;
  }

  // TODO: our sub-sub-device representation is currently [Level-Zero device
  // handle + Level-Zero compute group/engine index], so there is now no 1:1
  // mapping from L0 device handle to PI device assumed in this function. Until
  // Level-Zero adds unique ze_device_handle_t for sub-sub-devices, here we
  // filter out PI sub-sub-devices.
  std::shared_lock<pi_shared_mutex> Lock(PiDevicesCacheMutex);
  auto it = std::find_if(PiDevicesCache.begin(), PiDevicesCache.end(),
                         [&](std::unique_ptr<_zer_device_handle_t> &D) {
                           return D.get()->ZeDevice == ZeDevice &&
                                  (D.get()->RootDevice == nullptr ||
                                   D.get()->RootDevice->RootDevice == nullptr);
                         });
  if (it != PiDevicesCache.end()) {
    return (*it).get();
  }
  return nullptr;
}

// Check the device cache and load it if necessary.
zer_result_t _ur_platform_handle_t::populateDeviceCacheIfNeeded() {
  std::scoped_lock<pi_shared_mutex> Lock(PiDevicesCacheMutex);

  if (DeviceCachePopulated) {
    return ZER_RESULT_SUCCESS;
  }

  uint32_t ZeDeviceCount = 0;
  ZE_CALL(zeDeviceGet, (ZeDriver, &ZeDeviceCount, nullptr));

  try {
    std::vector<ze_device_handle_t> ZeDevices(ZeDeviceCount);
    ZE_CALL(zeDeviceGet, (ZeDriver, &ZeDeviceCount, ZeDevices.data()));

    for (uint32_t I = 0; I < ZeDeviceCount; ++I) {
      std::unique_ptr<_zer_device_handle_t> Device(
          new _zer_device_handle_t(ZeDevices[I], (zer_platform_handle_t)this));
      auto Result = Device->initialize();
      if (Result != ZER_RESULT_SUCCESS) {
        return Result;
      }

      // Additionally we need to cache all sub-devices too, such that they
      // are readily visible to the piextDeviceCreateWithNativeHandle.
      //
      uint32_t SubDevicesCount = 0;
      ZE_CALL(zeDeviceGetSubDevices,
              (Device->ZeDevice, &SubDevicesCount, nullptr));

      auto ZeSubdevices = new ze_device_handle_t[SubDevicesCount];
      ZE_CALL(zeDeviceGetSubDevices,
              (Device->ZeDevice, &SubDevicesCount, ZeSubdevices));

      // Wrap the Level Zero sub-devices into PI sub-devices, and add them to
      // cache.
      for (uint32_t I = 0; I < SubDevicesCount; ++I) {
        std::unique_ptr<_zer_device_handle_t> PiSubDevice(
            new _zer_device_handle_t(
                ZeSubdevices[I], (zer_platform_handle_t)this, Device.get()));
        auto Result = PiSubDevice->initialize();
        if (Result != ZER_RESULT_SUCCESS) {
          delete[] ZeSubdevices;
          return Result;
        }

        // collect all the ordinals for the sub-sub-devices
        std::vector<int> Ordinals;

        uint32_t numQueueGroups = 0;
        ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
                (PiSubDevice->ZeDevice, &numQueueGroups, nullptr));
        if (numQueueGroups == 0) {
          return ZER_RESULT_ERROR_UNKNOWN;
        }
        std::vector<ze_command_queue_group_properties_t> QueueGroupProperties(
            numQueueGroups);
        ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
                (PiSubDevice->ZeDevice, &numQueueGroups,
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
        if (PiSubDevice->isPVC() || ExposeCSliceInAffinityPartitioning) {
          // Create PI sub-sub-devices with the sub-device for all the ordinals.
          // Each {ordinal, index} points to a specific CCS which constructs
          // a sub-sub-device at this point.
          //
          // FIXME: Level Zero creates multiple PiDevices for a single physical
          // device when sub-device is partitioned into sub-sub-devices.
          // Sub-sub-device is technically a command queue and we should not
          // build program for each command queue. PiDevice is probably not the
          // right abstraction for a Level Zero command queue.
          for (uint32_t J = 0; J < Ordinals.size(); ++J) {
            for (uint32_t K = 0;
                 K < QueueGroupProperties[Ordinals[J]].numQueues; ++K) {
              std::unique_ptr<_zer_device_handle_t> PiSubSubDevice(
                  new _zer_device_handle_t(ZeSubdevices[I],
                                           (zer_platform_handle_t)this,
                                           PiSubDevice.get()));
              auto Result = PiSubSubDevice->initialize(Ordinals[J], K);
              if (Result != ZER_RESULT_SUCCESS) {
                return Result;
              }

              // save pointers to sub-sub-devices for quick retrieval in the
              // future.
              PiSubDevice->SubDevices.push_back(PiSubSubDevice.get());
              PiDevicesCache.push_back(std::move(PiSubSubDevice));
            }
          }
        }

        // save pointers to sub-devices for quick retrieval in the future.
        Device->SubDevices.push_back(PiSubDevice.get());
        PiDevicesCache.push_back(std::move(PiSubDevice));
      }
      delete[] ZeSubdevices;

      // Save the root device in the cache for future uses.
      PiDevicesCache.push_back(std::move(Device));
    }
  } catch (const std::bad_alloc &) {
    return ZER_RESULT_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return ZER_RESULT_ERROR_UNKNOWN;
  }
  DeviceCachePopulated = true;
  return ZER_RESULT_SUCCESS;
}

zer_result_t zerDeviceGetReference(zer_device_handle_t Device) {
  PI_ASSERT(Device, ZER_RESULT_INVALID_DEVICE);

  // The root-device ref-count remains unchanged (always 1).
  if (Device->isSubDevice()) {
    Device->RefCount.increment();
  }
  return ZER_RESULT_SUCCESS;
}

zer_result_t zerDeviceRelease(zer_device_handle_t Device) {
  PI_ASSERT(Device, ZER_RESULT_INVALID_DEVICE);

  // Root devices are destroyed during the piTearDown process.
  if (Device->isSubDevice()) {
    if (Device->RefCount.decrementAndTest()) {
      delete Device;
    }
  }

  return ZER_RESULT_SUCCESS;
}

ZER_APIEXPORT zer_result_t ZER_APICALL zerDevicePartition(
    zer_device_handle_t Device, ///< [in] handle of the device to partition.
    zer_device_partition_property_value_t
        *Properties, ///< [in] null-terminated array of <property, value> pair
                     ///< of the requested partitioning.
    uint32_t
        *NumDevices, ///< [in,out] pointer to the number of sub-devices.
                     ///< If count is zero, then the function shall update the
                     ///< value with the total number of sub-devices available.
                     ///< If count is greater than the number of sub-devices
                     ///< available, then the function shall update the value
                     ///< with the correct number of sub-devices available.
    zer_device_handle_t
        *OutDevices ///< [out][optional][range(0, *pCount)] array of handle of
                    ///< devices. If count is less than the number of
                    ///< sub-devices available, then the function shall only
                    ///< retrieve that number of sub-devices.
) {
  PI_ASSERT(NumDevices, ZER_RESULT_INVALID_VALUE);
  PI_ASSERT(Device, ZER_RESULT_INVALID_DEVICE);
  // Other partitioning ways are not supported by Level Zero
  if (Properties->property ==
      ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN) {
    if ((Properties->value !=
             ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE &&
         Properties->value != ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA))
      return ZER_RESULT_INVALID_VALUE;
  } else if (Properties->property ==
             ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE) {
    if (Properties->value != 0)
      return ZER_RESULT_INVALID_VALUE;
  } else {
    return ZER_RESULT_INVALID_VALUE;
  }

  // Devices cache is normally created in piDevicesGet but still make
  // sure that cache is populated.
  //
  auto Res = Device->Platform->populateDeviceCacheIfNeeded();
  if (Res != ZER_RESULT_SUCCESS) {
    return Res;
  }

  auto EffectiveNumDevices = [&]() -> decltype(Device->SubDevices.size()) {
    if (Device->SubDevices.size() == 0)
      return 0;

    // Sub-Sub-Devices are partitioned by CSlices, not by affinity domain.
    // However, if
    // SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING overrides that
    // still expose CSlices in partitioning by affinity domain for compatibility
    // reasons.
    if (Properties->property ==
            ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN &&
        !ExposeCSliceInAffinityPartitioning) {
      if (Device->isSubDevice())
        return 0;
    }
    if (Properties->property ==
        ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE) {
      // Not a CSlice-based partitioning.
      if (!Device->SubDevices[0]->isCCS())
        return 0;
    }

    return Device->SubDevices.size();
  }();

  if (*NumDevices) {
    // TODO: Consider support for partitioning to <= total sub-devices.
    // Currently supported partitioning (by affinity domain/numa) would always
    // partition to all sub-devices.
    //
    PI_ASSERT(*NumDevices == EffectiveNumDevices, ZER_RESULT_INVALID_VALUE);

    for (uint32_t I = 0; I < *NumDevices; I++) {
      OutDevices[I] = Device->SubDevices[I];
      // reusing the same pi_device needs to increment the reference count
      zerDeviceGetReference(OutDevices[I]);
    }
  }
  *NumDevices = EffectiveNumDevices;
  return ZER_RESULT_SUCCESS;
}
