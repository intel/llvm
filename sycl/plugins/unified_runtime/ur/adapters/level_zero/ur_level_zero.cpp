//===--------- ur_level_zero.hpp - Level Zero Adapter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <algorithm>
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
