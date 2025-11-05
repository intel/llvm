//===--------- adapter.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "adapter.hpp"
#include "common.hpp"
#include "ur_level_zero.hpp"
#include <iomanip>

// As windows order of unloading dlls is reversed from linux, windows will call
// umfTearDown before it could release umf objects in level_zero, so we call
// umfInit on urAdapterGet and umfAdapterTearDown to enforce the teardown of umf
// after umf objects are destructed.
#if defined(_WIN32)
#include <umf.h>
#endif

ZeUSMImportExtension ZeUSMImport;

// Due to multiple DLLMain definitions with SYCL, Global Adapter is init at
// variable creation.
#if defined(_WIN32)
ur_adapter_handle_t_ *GlobalAdapter = new ur_adapter_handle_t_();
#else
ur_adapter_handle_t_ *GlobalAdapter;
#endif
// This is a temporary workaround on windows, where UR adapter is teardowned
// before the UR loader, which will result in access violation when we use print
// function as the overrided print function was already released with the UR
// adapter.
// TODO: Change adapters to use a common sink class in the loader instead of
// using thier own sink class that inherit from logger::Sink.
class ur_legacy_sink : public logger::Sink {
public:
  ur_legacy_sink(std::string logger_name = "", bool skip_prefix = true)
      : Sink(std::move(logger_name), skip_prefix) {
    this->ostream = &std::cerr;
  }

  virtual void print([[maybe_unused]] ur_logger_level_t level,
                     const std::string &msg) override {
    fprintf(stderr, "%s", msg.c_str());
  }

  ~ur_legacy_sink() {
#if defined(_WIN32)
    logger::isTearDowned = true;
#endif
  };
};

// Find the corresponding ZesDevice Handle for a given ZeDevice
ur_result_t getZesDeviceHandle(ur_adapter_handle_t_ *adapter,
                               zes_uuid_t coreDeviceUuid,
                               zes_device_handle_t *ZesDevice,
                               uint32_t *SubDeviceId, ze_bool_t *SubDevice) {
  uint32_t ZesDriverCount = 0;
  std::vector<zes_driver_handle_t> ZesDrivers;
  std::vector<zes_device_handle_t> ZesDevices;
  ze_result_t ZesResult = ZE_RESULT_ERROR_INVALID_ARGUMENT;
  ZE2UR_CALL(adapter->getSysManDriversFunctionPtr, (&ZesDriverCount, nullptr));
  ZesDrivers.resize(ZesDriverCount);
  ZE2UR_CALL(adapter->getSysManDriversFunctionPtr,
             (&ZesDriverCount, ZesDrivers.data()));
  for (uint32_t I = 0; I < ZesDriverCount; ++I) {
    ZesResult = ZE_CALL_NOCHECK(
        adapter->getDeviceByUUIdFunctionPtr,
        (ZesDrivers[I], coreDeviceUuid, ZesDevice, SubDevice, SubDeviceId));
    if (ZesResult == ZE_RESULT_SUCCESS) {
      return UR_RESULT_SUCCESS;
    }
  }
  return UR_RESULT_ERROR_INVALID_ARGUMENT;
}

ur_result_t checkDeviceIntelGPUIpVersionOrNewer(uint32_t ipVersion) {
  uint32_t ZeDriverCount = 0;
  ZE2UR_CALL(zeDriverGet, (&ZeDriverCount, nullptr));
  if (ZeDriverCount == 0) {
    return UR_RESULT_SUCCESS;
  }

  std::vector<ze_driver_handle_t> ZeDrivers;
  std::vector<ze_device_handle_t> ZeDevices;
  ZeDrivers.resize(ZeDriverCount);

  ZE2UR_CALL(zeDriverGet, (&ZeDriverCount, ZeDrivers.data()));
  for (uint32_t I = 0; I < ZeDriverCount; ++I) {
    ze_device_properties_t device_properties{};
    ze_device_ip_version_ext_t ipVersionExt{};
    ipVersionExt.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;
    ipVersionExt.pNext = nullptr;
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = &ipVersionExt;
    uint32_t ZeDeviceCount = 0;
    ZE2UR_CALL(zeDeviceGet, (ZeDrivers[I], &ZeDeviceCount, nullptr));
    ZeDevices.resize(ZeDeviceCount);
    ZE2UR_CALL(zeDeviceGet, (ZeDrivers[I], &ZeDeviceCount, ZeDevices.data()));
    // Check if this driver has GPU Devices that have this IP Version or newer.
    for (uint32_t D = 0; D < ZeDeviceCount; ++D) {
      ZE2UR_CALL(zeDeviceGetProperties, (ZeDevices[D], &device_properties));
      if (device_properties.type == ZE_DEVICE_TYPE_GPU &&
          device_properties.vendorId == 0x8086) {
        ze_device_ip_version_ext_t *ipVersionExt =
            (ze_device_ip_version_ext_t *)device_properties.pNext;
        if (ipVersionExt->ipVersion >= ipVersion) {
          return UR_RESULT_SUCCESS;
        }
      }
    }
  }
  return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
}

/**
 * @brief Initializes the platforms by querying Level Zero drivers and devices.
 *
 * This function initializes the platforms by querying the available Level Zero
 * drivers and devices. It handles different behaviors based on the presence of
 * drivers obtained through `zeDriverGet` and initialized drivers through
 * `zeInitDrivers`.
 *
 * @param platforms A vector to store the initialized platform handles.
 * @param ZesResult The result of a previous ZES (Level Zero System) operation.
 * @return ur_result_t The result of the initialization process.
 *
 * The function performs the following steps:
 * 1. Queries the number of Level Zero drivers using `zeDriverGet`.
 * 2. If drivers are found, it retrieves their handles.
 * 3. If no drivers are found in either `zeInitDrivers` or `zeDriverGet`,
 *    it logs a message and returns success.
 * 4. If `zeInitDrivers` is supported by the global adapter, it retrieves
 *    their handles and properties.
 * 5. It compares the drivers obtained from `zeDriverGet` and `zeInitDrivers`,
 *    adding unique drivers to the list.
 * 6. If `zeInitDrivers` is not supported, it uses the drivers obtained
 *    from `zeDriverGet`.
 * 7. For each driver, it queries the devices and checks if they are GPU
 * devices.
 * 8. If a GPU device is found, it initializes a platform for the driver and
 *    adds it to the platforms vector.
 * 9. If ZES operations are successful, it populates the ZES/ZE device mapping
 *    for the devices into the platform.
 * 10. The function handles exceptions and returns the appropriate result.
 */
ur_result_t initPlatforms(ur_adapter_handle_t_ *adapter, PlatformVec &platforms,
                          ze_result_t ZesResult) noexcept try {
  std::vector<ze_driver_handle_t> ZeDrivers;
  std::vector<ze_driver_handle_t> ZeDriverGetHandles;
  std::vector<ze_driver_handle_t> ZeInitDriversHandles;
  std::vector<ze_device_handle_t> ZeDevices;
  uint32_t ZeDriverCount = 0;
  uint32_t ZeDriverGetCount = 0;

  auto ZeDriverGetResult =
      ZE_CALL_NOCHECK(zeDriverGet, (&ZeDriverGetCount, nullptr));
  if (ZeDriverGetCount > 0 && ZeDriverGetResult == ZE_RESULT_SUCCESS) {
    ZeDriverGetHandles.resize(ZeDriverGetCount);
    ZE2UR_CALL(zeDriverGet, (&ZeDriverGetCount, ZeDriverGetHandles.data()));
  }
  if (ZeDriverGetCount == 0 && adapter->ZeInitDriversCount == 0) {
    UR_LOG(ERR, "\nNo Valid L0 Drivers found.\n");
    return UR_RESULT_SUCCESS;
  }

  if (adapter->InitDriversSupported) {
    ZeInitDriversHandles.resize(adapter->ZeInitDriversCount);
    ZeDrivers.resize(adapter->ZeInitDriversCount);
    ZE2UR_CALL(adapter->initDriversFunctionPtr,
               (&adapter->ZeInitDriversCount, ZeInitDriversHandles.data(),
                &adapter->InitDriversDesc));
    ZeDrivers.assign(ZeInitDriversHandles.begin(), ZeInitDriversHandles.end());
    if (ZeDriverGetCount > 0 && adapter->ZeInitDriversCount > 0) {
      for (uint32_t X = 0; X < adapter->ZeInitDriversCount; ++X) {
        // zeDriverGet and zeInitDrivers can return the driver handles in
        // reverse order based on driver ordering causing an issue if the
        // drivers are expected to be in the same order. To resolve, this loop
        // checks if the driver from zeDriverGet already exists in zeInitDrivers
        // and only adds it if it is not found.
        bool unMatchedDriverHandle = false;
        ze_driver_handle_t driverGetHandle = nullptr;
        for (uint32_t Y = 0; Y < ZeDriverGetCount; ++Y) {
          unMatchedDriverHandle = true;
          ZeStruct<ze_driver_properties_t> ZeDriverGetProperties;
          ZeStruct<ze_driver_properties_t> ZeInitDriverProperties;
          ZE2UR_CALL(zeDriverGetProperties,
                     (ZeDriverGetHandles[Y], &ZeDriverGetProperties));
          ZE2UR_CALL(zeDriverGetProperties,
                     (ZeInitDriversHandles[X], &ZeInitDriverProperties));
          driverGetHandle = ZeDriverGetHandles[Y];
          // If zeDriverGet driver is the same version as zeInitDriver driver,
          // then do not add it again.
          if (ZeDriverGetProperties.driverVersion ==
              ZeInitDriverProperties.driverVersion) {
            UR_LOG(DEBUG,
                   "\nzeDriverHandle {} matched between zeDriverGet and "
                   "zeInitDrivers. Not adding duplicate driver to list\n",
                   ZeDriverGetHandles[Y]);
            unMatchedDriverHandle = false;
            break;
          }
        }
        if (unMatchedDriverHandle) {
          UR_LOG(DEBUG,
                 "\nzeDriverHandle {} not found in zeInitDrivers. Adding to "
                 "driver list.\n",
                 driverGetHandle);
          ZeDrivers.push_back(driverGetHandle);
        }
      }
    }
  } else {
    ZeDrivers.resize(ZeDriverGetCount);
    ZeDrivers.assign(ZeDriverGetHandles.begin(), ZeDriverGetHandles.end());
  }
  ZeDriverCount = ZeDrivers.size();
  UR_LOG(DEBUG, "\n{} L0 Drivers found.\n", ZeDriverCount);
  for (uint32_t I = 0; I < ZeDriverCount; ++I) {
    // Keep track of the first platform init for this Driver
    bool DriverPlatformInit = false;
    ze_device_properties_t device_properties{};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    uint32_t ZeDeviceCount = 0;
    ZE2UR_CALL(zeDeviceGet, (ZeDrivers[I], &ZeDeviceCount, nullptr));
    ZeDevices.resize(ZeDeviceCount);
    ZE2UR_CALL(zeDeviceGet, (ZeDrivers[I], &ZeDeviceCount, ZeDevices.data()));
    auto platform = std::make_unique<ur_platform_handle_t_>(ZeDrivers[I]);
    // Check if this driver has GPU Devices
    for (uint32_t D = 0; D < ZeDeviceCount; ++D) {
      ZE2UR_CALL(zeDeviceGetProperties, (ZeDevices[D], &device_properties));
      if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
        // Check if this driver's platform has already been init.
        if (!DriverPlatformInit) {
          // If this Driver is a GPU, save it as a usable platform.
          UR_CALL(platform->initialize());

          // Save a copy in the cache for future uses.
          platforms.push_back(std::move(platform));
          // Mark this driver's platform as init to prevent additional platforms
          // from being created per driver.
          DriverPlatformInit = true;
        }
        if (ZesResult == ZE_RESULT_SUCCESS) {
          // Populate the Zes/Ze device mapping for this Ze Device into the last
          // added platform which represents the current driver being queried.
          ur_zes_device_handle_data_t ZesDeviceData;
          zes_uuid_t ZesUUID;
          std::memcpy(&ZesUUID, &device_properties.uuid, sizeof(zes_uuid_t));
          if (getZesDeviceHandle(adapter, ZesUUID, &ZesDeviceData.ZesDevice,
                                 &ZesDeviceData.SubDeviceId,
                                 &ZesDeviceData.SubDevice) ==
              UR_RESULT_SUCCESS) {
            platforms.back()->ZedeviceToZesDeviceMap.insert(
                std::make_pair(ZeDevices[D], std::move(ZesDeviceData)));
          }
        }
      }
    }
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t adapterStateInit() {

#if defined(_WIN32)
  umfInit();
#endif

  return UR_RESULT_SUCCESS;
}

static bool isBMGorNewer() {
  auto urResult = checkDeviceIntelGPUIpVersionOrNewer(0x05004000);
  if (urResult != UR_RESULT_SUCCESS &&
      urResult != UR_RESULT_ERROR_UNSUPPORTED_VERSION) {
    UR_LOG(ERR, "Intel GPU IP Version check failed: {}\n", urResult);
    throw urResult;
  }

  return urResult == UR_RESULT_SUCCESS;
}

// returns a pair indicating whether to use the V1 adapter and a string
// indicating the reason for the decision.
static std::pair<bool, std::string> shouldUseV1Adapter() {
  auto specificAdapterVersionRequested =
      ur_getenv("UR_LOADER_USE_LEVEL_ZERO_V2").has_value() ||
      ur_getenv("SYCL_UR_USE_LEVEL_ZERO_V2").has_value();

  auto v2Requested = getenv_tobool("UR_LOADER_USE_LEVEL_ZERO_V2", false);
  v2Requested |= getenv_tobool("SYCL_UR_USE_LEVEL_ZERO_V2", false);

  std::string reason =
      specificAdapterVersionRequested
          ? "Specific adapter version requested by UR_LOADER_USE_LEVEL_ZERO_V2 "
            "or SYCL_UR_USE_LEVEL_ZERO_V2"
          : "Using default adapter version based on device IP version";

  if (v2Requested) {
    return {false, reason};
  }

  if (!v2Requested && specificAdapterVersionRequested) {
    // v1 specifically requested
    return {true, reason};
  }

  // default: only enable for devices older than BMG
  return {!isBMGorNewer(), reason};
}

[[maybe_unused]] static std::pair<bool, std::string> shouldUseV2Adapter() {
  auto [useV1, reason] = shouldUseV1Adapter();
  return {!useV1, reason};
}

/*
This constructor initializes the `ur_adapter_handle_t_` object and
sets up the environment for Level Zero (L0) initialization.
The behavior of the initialization process is influenced by two
environment variables:
`UR_L0_ENABLE_SYSMAN_ENV_DEFAULT` and `UR_L0_ENABLE_ZESINIT_DEFAULT`.

| Environment Variable           | Value | Behavior                   |
|--------------------------------|-------|----------------------------|
| UR_L0_ENABLE_SYSMAN_ENV_DEFAULT| 1     | Enables the default SysMan |
|                                |       | environment initialization |
|                                |       | by setting                 |
|                                |       | `ZES_ENABLE_SYSMAN` to "1".|
|                                | 0     | Disables the default SysMan|
|                                |       | environment initialization.|
|                                | unset | Defaults to 1, enabling the|
|                                |       | SysMan environment         |
|                                |       | initialization.            |
| UR_L0_ENABLE_ZESINIT_DEFAULT   | 1     | Enables the default SysMan |
|                                |       | initialization by loading  |
|                                |       | SysMan-related functions   |
|                                |       | and calling `zesInit`.     |
|                                | 0     | Disables the default SysMan|
|                                |       | initialization with zesInit|
|                                | unset | Defaults to 0, disabling   |
|                                |       | the SysMan initialization  |
|                                |       | thru zesInit.              |

Behavior Summary:
- If `UR_L0_ENABLE_SYSMAN_ENV_DEFAULT` is set to 1 or is unset,
  `ZES_ENABLE_SYSMAN` is set to "1".
- If `UR_L0_ENABLE_ZESINIT_DEFAULT` is set to 1 and
  `UR_L0_ENABLE_SYSMAN_ENV_DEFAULT` is not set to 1,
  SysMan-related functions are loaded and `zesInit` is called.
- If `UR_L0_ENABLE_ZESINIT_DEFAULT` is set to 0 or is unset,
  SysMan initialization is skipped.
*/
ur_adapter_handle_t_::ur_adapter_handle_t_()
    : handle_base(), logger(logger::get_logger("level_zero")), RefCount(0) {
  auto ZeInitDriversResult = ZE_RESULT_ERROR_UNINITIALIZED;
  auto ZeInitResult = ZE_RESULT_ERROR_UNINITIALIZED;
  auto ZesResult = ZE_RESULT_ERROR_UNINITIALIZED;
  auto SyclUrTrace = ur_getenv("SYCL_UR_TRACE");

#ifdef UR_STATIC_LEVEL_ZERO
  // Given static linking of the L0 Loader, we must delay the loader's
  // destruction of its context until after the UR Adapter is destroyed.
  zelSetDelayLoaderContextTeardown();
#endif

  if (UrL0Debug & UR_L0_DEBUG_BASIC) {
    logger.setLegacySink(std::make_unique<ur_legacy_sink>());
    setEnvVar("ZEL_ENABLE_LOADER_LOGGING", "1");
    setEnvVar("ZEL_LOADER_LOGGING_LEVEL", "trace");
    setEnvVar("ZEL_LOADER_LOG_CONSOLE", "1");
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
  }
#if defined(_WIN32)
  else if (SyclUrTrace.has_value()) {
    logger.setLegacySink(std::make_unique<ur_legacy_sink>());
  }
#endif

  if (UrL0Debug & UR_L0_DEBUG_VALIDATION) {
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
    setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
  }

  if (UrL0LeaksDebug) {
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
    setEnvVar("ZEL_ENABLE_BASIC_LEAK_CHECKER", "1");
  }

  uint32_t UserForcedSysManInit = 0;
  // Check if the user has disabled the default L0 Env initialization.
  const int UrSysManEnvInitEnabled = [&UserForcedSysManInit] {
    const char *UrRet = std::getenv("UR_L0_ENABLE_SYSMAN_ENV_DEFAULT");
    if (!UrRet)
      return 1;
    UserForcedSysManInit &= 1;
    return std::atoi(UrRet);
  }();

  // Dynamically load the new L0 apis separately.
  // This must be done to avoid attempting to use symbols that do
  // not exist in older loader runtimes.
#ifndef UR_STATIC_LEVEL_ZERO
#ifdef _WIN32
  processHandle = GetModuleHandle(NULL);
#else
  processHandle = nullptr;
#endif
#endif

  // Setting these environment variables before running zeInit will enable
  // the validation layer in the Level Zero loader.
  if (UrL0Debug & UR_L0_DEBUG_VALIDATION) {
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
    setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
  }

  if (getenv("SYCL_ENABLE_PCI") != nullptr) {
    UR_LOG(WARN,
           "WARNING: SYCL_ENABLE_PCI is deprecated and no longer needed.\n");
  }

  // TODO: We can still safely recover if something goes wrong during the
  // init. Implement handling segfault using sigaction.

  // We must only initialize the driver once, even if urPlatformGet() is
  // called multiple times.  Declaring the return value as "static" ensures
  // it's only called once.

  // Set ZES_ENABLE_SYSMAN by default if the user has not set it.
  if (UrSysManEnvInitEnabled) {
    setEnvVar("ZES_ENABLE_SYSMAN", "1");
  }

  // Init with all flags set to enable for all driver types to be init in
  // the application.
  ze_init_flags_t L0InitFlags = ZE_INIT_FLAG_GPU_ONLY;
  if (UrL0InitAllDrivers) {
    L0InitFlags |= ZE_INIT_FLAG_VPU_ONLY;
  }
  UR_LOG(DEBUG, "\nzeInit with flags value of {}\n",
         static_cast<int>(L0InitFlags));
  ZeInitResult = ZE_CALL_NOCHECK(zeInit, (L0InitFlags));
  if (ZeInitResult != ZE_RESULT_SUCCESS) {
    const char *ErrorString = "Unknown";
    zeParseError(ZeInitResult, ErrorString);
    UR_LOG(ERR, "\nzeInit failed with {}\n", ErrorString);
  }

  bool useInitDrivers = false;
  zel_version_t loader_version = {};
  size_t num_components;
  auto result = zelLoaderGetVersions(&num_components, nullptr);
  if (result == ZE_RESULT_SUCCESS) {
    zel_component_version_t *versions =
        new zel_component_version_t[num_components];
    result = zelLoaderGetVersions(&num_components, versions);
    if (result == ZE_RESULT_SUCCESS) {
      for (size_t i = 0; i < num_components; ++i) {
        if (strncmp(versions[i].component_name, "loader", strlen("loader")) ==
            0) {
          loader_version = versions[i].component_lib_version;
          UR_LOG(DEBUG, "\nLevel Zero Loader Version: {}.{}.{}\n",
                 loader_version.major, loader_version.minor,
                 loader_version.patch);
          break;
        }
      }
    }
    delete[] versions;
    if (loader_version.major > 1 ||
        (loader_version.major == 1 && loader_version.minor > 19) ||
        (loader_version.major == 1 && loader_version.minor == 19 &&
         loader_version.patch >= 2)) {
      useInitDrivers = true;
    }

    if ((loader_version.major == 1 && loader_version.minor < 21) ||
        (loader_version.major == 1 && loader_version.minor == 21 &&
         loader_version.patch < 2)) {
      UR_LOG(WARN,
             "WARNING: Level Zero Loader version is older than 1.21.2. "
             "Please update to the latest version for API logging support.\n");
    }
  }

  if (useInitDrivers) {
#ifdef UR_STATIC_LEVEL_ZERO
    initDriversFunctionPtr = zeInitDrivers;
#else
    initDriversFunctionPtr =
        (ze_pfnInitDrivers_t)ur_loader::LibLoader::getFunctionPtr(
            processHandle, "zeInitDrivers");
#endif
    if (initDriversFunctionPtr) {
      UR_LOG(DEBUG, "\nzeInitDrivers with flags value of {}\n",
             static_cast<int>(InitDriversDesc.flags));
      ZeInitDriversResult =
          ZE_CALL_NOCHECK(initDriversFunctionPtr,
                          (&ZeInitDriversCount, nullptr, &InitDriversDesc));
      if (ZeInitDriversResult == ZE_RESULT_SUCCESS) {
        InitDriversSupported = true;
      } else {
        const char *ErrorString = "Unknown";
        zeParseError(ZeInitDriversResult, ErrorString);
        UR_LOG(ERR, "\nzeInitDrivers failed with {}\n", ErrorString);
      }
    }
  }

  if (ZeInitResult != ZE_RESULT_SUCCESS &&
      ZeInitDriversResult != ZE_RESULT_SUCCESS) {
    // Absorb the ZE_RESULT_ERROR_UNINITIALIZED and just return 0 Platforms.
    UR_LOG(ERR, "Level Zero Uninitialized\n");
    return;
  }

  PlatformVec platforms;

  bool forceLoadedAdapter = ur_getenv("UR_ADAPTERS_FORCE_LOAD").has_value();
  if (!forceLoadedAdapter) {
#ifdef UR_ADAPTER_LEVEL_ZERO_V2
    auto [useV2, reason] = shouldUseV2Adapter();
    if (!useV2) {
      UR_LOG(INFO, "Skipping L0 V2 adapter: {}", reason);
      return;
    }
#else
    auto [useV1, reason] = shouldUseV1Adapter();
    if (!useV1) {
      UR_LOG(INFO, "Skipping L0 V1 adapter: {}", reason);
      return;
    }
#endif
  }

  // Check if the user has enabled the default L0 SysMan initialization.
  const int UrSysmanZesinitEnable = [&UserForcedSysManInit] {
    const char *UrRet = std::getenv("UR_L0_ENABLE_ZESINIT_DEFAULT");
    if (!UrRet)
      return 0;
    UserForcedSysManInit &= 2;
    return std::atoi(UrRet);
  }();

  bool ZesInitNeeded = UrSysmanZesinitEnable && !UrSysManEnvInitEnabled;
  // Unless the user has forced the SysMan init, we will check the device
  // version to see if the zesInit is needed.
  if (UserForcedSysManInit == 0 &&
      checkDeviceIntelGPUIpVersionOrNewer(0x05004000) == UR_RESULT_SUCCESS) {
    if (UrSysManEnvInitEnabled) {
      setEnvVar("ZES_ENABLE_SYSMAN", "0");
    }
    ZesInitNeeded = true;
  }
  if (ZesInitNeeded) {
#ifdef UR_STATIC_LEVEL_ZERO
    getDeviceByUUIdFunctionPtr = zesDriverGetDeviceByUuidExp;
    getSysManDriversFunctionPtr = zesDriverGet;
    sysManInitFunctionPtr = zesInit;
#else
    getDeviceByUUIdFunctionPtr =
        (zes_pfnDriverGetDeviceByUuidExp_t)ur_loader::LibLoader::getFunctionPtr(
            processHandle, "zesDriverGetDeviceByUuidExp");
    getSysManDriversFunctionPtr =
        (zes_pfnDriverGet_t)ur_loader::LibLoader::getFunctionPtr(
            processHandle, "zesDriverGet");
    sysManInitFunctionPtr = (zes_pfnInit_t)ur_loader::LibLoader::getFunctionPtr(
        processHandle, "zesInit");
#endif
  }
  if (getDeviceByUUIdFunctionPtr && getSysManDriversFunctionPtr &&
      sysManInitFunctionPtr) {
    ze_init_flags_t L0ZesInitFlags = 0;
    UR_LOG(DEBUG, "\nzesInit with flags value of {}\n",
           static_cast<int>(L0ZesInitFlags));
    ZesResult = ZE_CALL_NOCHECK(sysManInitFunctionPtr, (L0ZesInitFlags));
  } else {
    ZesResult = ZE_RESULT_ERROR_UNINITIALIZED;
  }

  ur_result_t err = initPlatforms(this, platforms, ZesResult);
  if (err == UR_RESULT_SUCCESS) {
    Platforms = std::move(platforms);
  } else {
    UR_LOG(ERR, "Failed to initialize Platforms");
    throw err;
  }
}

void globalAdapterOnDemandCleanup() {
  if (GlobalAdapter) {
    delete GlobalAdapter;
  }
}

ur_result_t adapterStateTeardown() {
  // Due to multiple DLLMain definitions with SYCL, register to cleanup the
  // Global Adapter after refcnt is 0
#if defined(_WIN32)
  umfTearDown();
  std::atexit(globalAdapterOnDemandCleanup);
#endif

  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {
ur_result_t urAdapterGet(
    /// [in] the number of platforms to be added to phAdapters. If phAdapters is
    /// not NULL, then NumEntries should be greater than zero, otherwise
    /// ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)] array of handle of adapters.
    /// If NumEntries is less than the number of adapters available, then
    /// ::urAdapterGet shall only retrieve that number of platforms.
    ur_adapter_handle_t *Adapters,
    /// [out][optional] returns the total number of adapters available.
    uint32_t *NumAdapters) try {
  static std::mutex AdapterConstructionMutex{};

  // We need to initialize the adapter even if user only queries
  // the number of adapters to decided whether to use V1 or V2.
  std::lock_guard<std::mutex> Lock{AdapterConstructionMutex};

  if (!GlobalAdapter) {
    GlobalAdapter = new ur_adapter_handle_t_();
  }

  if (GlobalAdapter->Platforms.size() == 0) {
    if (NumAdapters) {
      *NumAdapters = 0;
    }
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  if (NumEntries && Adapters) {
    *Adapters = GlobalAdapter;

    if (GlobalAdapter->RefCount.retain() == 0) {
      adapterStateInit();
    }
  }

  if (NumAdapters) {
    *NumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t result) {
  return result;
} catch (...) {
  return UR_RESULT_ERROR_UNKNOWN;
}

ur_result_t urAdapterRelease([[maybe_unused]] ur_adapter_handle_t Adapter) {
  assert(GlobalAdapter && GlobalAdapter == Adapter);

  // NOTE: This does not require guarding with a mutex; the instant the ref
  // count hits zero, both Get and Retain are UB.
  if (GlobalAdapter->RefCount.release()) {
    auto result = adapterStateTeardown();
#ifdef UR_STATIC_LEVEL_ZERO
    // Given static linking of the L0 Loader, we must delay the loader's
    // destruction of its context until after the UR Adapter is destroyed.
    zelLoaderContextTeardown();
#endif

    delete GlobalAdapter;
    GlobalAdapter = nullptr;

    return result;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterRetain([[maybe_unused]] ur_adapter_handle_t Adapter) {
  assert(GlobalAdapter && GlobalAdapter == Adapter);
  GlobalAdapter->RefCount.retain();

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterGetLastError(
    /// [in] handle of the platform instance
    ur_adapter_handle_t,
    /// [out] pointer to a C string where the adapter specific error message
    /// will be stored.
    const char **Message,
    /// [out] pointer to an integer where the adapter specific error code will
    /// be stored.
    int32_t *Error) {
  *Message = ErrorMessage;
  *Error = ErrorAdapterNativeCode;

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterGetInfo(ur_adapter_handle_t, ur_adapter_info_t PropName,
                             size_t PropSize, void *PropValue,
                             size_t *PropSizeRet) {
  UrReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case UR_ADAPTER_INFO_BACKEND:
    return ReturnValue(UR_BACKEND_LEVEL_ZERO);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(GlobalAdapter->RefCount.getCount());
  case UR_ADAPTER_INFO_VERSION: {
#ifdef UR_ADAPTER_LEVEL_ZERO_V2
    uint32_t adapterVersion = 2;
#else
    uint32_t adapterVersion = 1;
#endif
    return ReturnValue(adapterVersion);
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterSetLoggerCallback(
    ur_adapter_handle_t, ur_logger_callback_t pfnLoggerCallback,
    void *pUserData, ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET) {

  if (GlobalAdapter) {
    GlobalAdapter->logger.setCallbackSink(pfnLoggerCallback, pUserData, level);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterSetLoggerCallbackLevel(ur_adapter_handle_t,
                                            ur_logger_level_t level) {

  if (GlobalAdapter) {
    GlobalAdapter->logger.setCallbackLevel(level);
  }

  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
