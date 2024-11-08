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
#include "ur_level_zero.hpp"
#include <iomanip>

// As windows order of unloading dlls is reversed from linux, windows will call
// umfTearDown before it could release umf objects in level_zero, so we call
// umfInit on urAdapterGet and umfAdapterTearDown to enforce the teardown of umf
// after umf objects are destructed.
#if defined(_WIN32)
#include <umf.h>
#endif

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

  virtual void print([[maybe_unused]] logger::Level level,
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
ur_result_t getZesDeviceHandle(zes_uuid_t coreDeviceUuid,
                               zes_device_handle_t *ZesDevice,
                               uint32_t *SubDeviceId, ze_bool_t *SubDevice) {
  uint32_t ZesDriverCount = 0;
  std::vector<zes_driver_handle_t> ZesDrivers;
  std::vector<zes_device_handle_t> ZesDevices;
  ze_result_t ZesResult = ZE_RESULT_ERROR_INVALID_ARGUMENT;
  ZE2UR_CALL(GlobalAdapter->getSysManDriversFunctionPtr,
             (&ZesDriverCount, nullptr));
  ZesDrivers.resize(ZesDriverCount);
  ZE2UR_CALL(GlobalAdapter->getSysManDriversFunctionPtr,
             (&ZesDriverCount, ZesDrivers.data()));
  for (uint32_t I = 0; I < ZesDriverCount; ++I) {
    ZesResult = ZE_CALL_NOCHECK(
        GlobalAdapter->getDeviceByUUIdFunctionPtr,
        (ZesDrivers[I], coreDeviceUuid, ZesDevice, SubDevice, SubDeviceId));
    if (ZesResult == ZE_RESULT_SUCCESS) {
      return UR_RESULT_SUCCESS;
    }
  }
  return UR_RESULT_ERROR_INVALID_ARGUMENT;
}

ur_result_t initPlatforms(PlatformVec &platforms,
                          ze_result_t ZesResult) noexcept try {
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
          if (getZesDeviceHandle(
                  ZesUUID, &ZesDeviceData.ZesDevice, &ZesDeviceData.SubDeviceId,
                  &ZesDeviceData.SubDevice) == UR_RESULT_SUCCESS) {
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
    : logger(logger::get_logger("level_zero")) {

  if (UrL0Debug & UR_L0_DEBUG_BASIC) {
    logger.setLegacySink(std::make_unique<ur_legacy_sink>());
  };

  if (UrL0Debug & UR_L0_DEBUG_VALIDATION) {
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
    setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
  }

  PlatformCache.Compute = [](Result<PlatformVec> &result) {
    static std::once_flag ZeCallCountInitialized;
    try {
      std::call_once(ZeCallCountInitialized, []() {
        if (UrL0LeaksDebug) {
          ZeCallCount = new std::map<std::string, int>;
        }
      });
    } catch (...) {
      result = exceptionToResult(std::current_exception());
      return;
    }

    // Check if the user has disabled the default L0 Env initialization.
    const int UrSysManEnvInitEnabled = [] {
      const char *UrRet = std::getenv("UR_L0_ENABLE_SYSMAN_ENV_DEFAULT");
      if (!UrRet)
        return 1;
      return std::atoi(UrRet);
    }();

    // initialize level zero only once.
    if (GlobalAdapter->ZeResult == std::nullopt) {
      // Setting these environment variables before running zeInit will enable
      // the validation layer in the Level Zero loader.
      if (UrL0Debug & UR_L0_DEBUG_VALIDATION) {
        setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
        setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
      }

      if (getenv("SYCL_ENABLE_PCI") != nullptr) {
        logger::warning(
            "WARNING: SYCL_ENABLE_PCI is deprecated and no longer needed.\n");
      }

      // TODO: We can still safely recover if something goes wrong during the
      // init. Implement handling segfault using sigaction.

      // We must only initialize the driver once, even if urPlatformGet() is
      // called multiple times.  Declaring the return value as "static" ensures
      // it's only called once.

      // Init with all flags set to enable for all driver types to be init in
      // the application.
      ze_init_flags_t L0InitFlags = ZE_INIT_FLAG_GPU_ONLY;
      if (UrL0InitAllDrivers) {
        L0InitFlags |= ZE_INIT_FLAG_VPU_ONLY;
      }

      // Set ZES_ENABLE_SYSMAN by default if the user has not set it.
      if (UrSysManEnvInitEnabled) {
        setEnvVar("ZES_ENABLE_SYSMAN", "1");
      }
      logger::debug("\nzeInit with flags value of {}\n",
                    static_cast<int>(L0InitFlags));
      GlobalAdapter->ZeResult = ZE_CALL_NOCHECK(zeInit, (L0InitFlags));
    }
    assert(GlobalAdapter->ZeResult !=
           std::nullopt); // verify that level-zero is initialized
    PlatformVec platforms;

    // Absorb the ZE_RESULT_ERROR_UNINITIALIZED and just return 0 Platforms.
    if (*GlobalAdapter->ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
      result = std::move(platforms);
      return;
    }
    if (*GlobalAdapter->ZeResult != ZE_RESULT_SUCCESS) {
      logger::error("zeInit: Level Zero initialization failure\n");
      result = ze2urResult(*GlobalAdapter->ZeResult);

      return;
    }
    // Dynamically load the new L0 SysMan separate init and new EXP apis
    // separately. This must be done to avoid attempting to use symbols that do
    // not exist in older loader runtimes.
#ifdef _WIN32
    HMODULE processHandle = GetModuleHandle(NULL);
#else
    HMODULE processHandle = nullptr;
#endif

    // Check if the user has enabled the default L0 SysMan initialization.
    const int UrSysmanZesinitEnable = [] {
      const char *UrRet = std::getenv("UR_L0_ENABLE_ZESINIT_DEFAULT");
      if (!UrRet)
        return 0;
      return std::atoi(UrRet);
    }();

    // Enable zesInit by default only if ZES_ENABLE_SYSMAN has not been set by
    // default and UrSysmanZesinitEnable is true.
    if (UrSysmanZesinitEnable && !UrSysManEnvInitEnabled) {
      GlobalAdapter->getDeviceByUUIdFunctionPtr =
          (zes_pfnDriverGetDeviceByUuidExp_t)
              ur_loader::LibLoader::getFunctionPtr(
                  processHandle, "zesDriverGetDeviceByUuidExp");
      GlobalAdapter->getSysManDriversFunctionPtr =
          (zes_pfnDriverGet_t)ur_loader::LibLoader::getFunctionPtr(
              processHandle, "zesDriverGet");
      GlobalAdapter->sysManInitFunctionPtr =
          (zes_pfnInit_t)ur_loader::LibLoader::getFunctionPtr(processHandle,
                                                              "zesInit");
    }
    if (GlobalAdapter->getDeviceByUUIdFunctionPtr &&
        GlobalAdapter->getSysManDriversFunctionPtr &&
        GlobalAdapter->sysManInitFunctionPtr) {
      ze_init_flags_t L0ZesInitFlags = 0;
      logger::debug("\nzesInit with flags value of {}\n",
                    static_cast<int>(L0ZesInitFlags));
      GlobalAdapter->ZesResult = ZE_CALL_NOCHECK(
          GlobalAdapter->sysManInitFunctionPtr, (L0ZesInitFlags));
    } else {
      GlobalAdapter->ZesResult = ZE_RESULT_ERROR_UNINITIALIZED;
    }

    ur_result_t err = initPlatforms(platforms, *GlobalAdapter->ZesResult);
    if (err == UR_RESULT_SUCCESS) {
      result = std::move(platforms);
    } else {
      result = err;
    }
  };
}

void globalAdapterOnDemandCleanup() {
  if (GlobalAdapter) {
    delete GlobalAdapter;
  }
}

ur_result_t adapterStateTeardown() {
  // Print the balance of various create/destroy native calls.
  // The idea is to verify if the number of create(+) and destroy(-) calls are
  // matched.
  if (ZeCallCount && (UrL0LeaksDebug) != 0) {
    bool LeakFound = false;
    // clang-format off
    //
    // The format of this table is such that each row accounts for a
    // specific type of objects, and all elements in the raw except the last
    // one are allocating objects of that type, while the last element is known
    // to deallocate objects of that type.
    //
    std::vector<std::vector<std::string>> CreateDestroySet = {
      {"zeContextCreate",      "zeContextDestroy"},
      {"zeCommandQueueCreate", "zeCommandQueueDestroy"},
      {"zeModuleCreate",       "zeModuleDestroy"},
      {"zeKernelCreate",       "zeKernelDestroy"},
      {"zeEventPoolCreate",    "zeEventPoolDestroy"},
      {"zeCommandListCreateImmediate", "zeCommandListCreate", "zeCommandListDestroy"},
      {"zeEventCreate",        "zeEventDestroy"},
      {"zeFenceCreate",        "zeFenceDestroy"},
      {"zeImageCreate",        "zeImageDestroy"},
      {"zeSamplerCreate",      "zeSamplerDestroy"},
      {"zeMemAllocDevice", "zeMemAllocHost", "zeMemAllocShared", "zeMemFree"},
    };

    // A sample output aimed below is this:
    // ------------------------------------------------------------------------
    //                zeContextCreate = 1     \--->        zeContextDestroy = 1
    //           zeCommandQueueCreate = 1     \--->   zeCommandQueueDestroy = 1
    //                 zeModuleCreate = 1     \--->         zeModuleDestroy = 1
    //                 zeKernelCreate = 1     \--->         zeKernelDestroy = 1
    //              zeEventPoolCreate = 1     \--->      zeEventPoolDestroy = 1
    //   zeCommandListCreateImmediate = 1     |
    //            zeCommandListCreate = 1     \--->    zeCommandListDestroy = 1  ---> LEAK = 1
    //                  zeEventCreate = 2     \--->          zeEventDestroy = 2
    //                  zeFenceCreate = 1     \--->          zeFenceDestroy = 1
    //                  zeImageCreate = 0     \--->          zeImageDestroy = 0
    //                zeSamplerCreate = 0     \--->        zeSamplerDestroy = 0
    //               zeMemAllocDevice = 0     |
    //                 zeMemAllocHost = 1     |
    //               zeMemAllocShared = 0     \--->               zeMemFree = 1
    //
    // clang-format on
    // TODO: use logger to print this messages
    std::cerr << "Check balance of create/destroy calls\n";
    std::cerr << "----------------------------------------------------------\n";
    std::stringstream ss;
    for (const auto &Row : CreateDestroySet) {
      int diff = 0;
      for (auto I = Row.begin(); I != Row.end();) {
        const char *ZeName = (*I).c_str();
        const auto &ZeCount = (*ZeCallCount)[*I];

        bool First = (I == Row.begin());
        bool Last = (++I == Row.end());

        if (Last) {
          ss << " \\--->";
          diff -= ZeCount;
        } else {
          diff += ZeCount;
          if (!First) {
            ss << " | ";
            std::cerr << ss.str() << "\n";
            ss.str("");
            ss.clear();
          }
        }
        ss << std::setw(30) << std::right << ZeName;
        ss << " = ";
        ss << std::setw(5) << std::left << ZeCount;
      }

      if (diff) {
        LeakFound = true;
        ss << " ---> LEAK = " << diff;
      }

      std::cerr << ss.str() << '\n';
      ss.str("");
      ss.clear();
    }

    ZeCallCount->clear();
    delete ZeCallCount;
    ZeCallCount = nullptr;
    if (LeakFound)
      return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  }

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
    uint32_t NumEntries, ///< [in] the number of platforms to be added to
                         ///< phAdapters. If phAdapters is not NULL, then
                         ///< NumEntries should be greater than zero, otherwise
                         ///< ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    ur_adapter_handle_t
        *Adapters, ///< [out][optional][range(0, NumEntries)] array of handle of
                   ///< adapters. If NumEntries is less than the number of
                   ///< adapters available, then
                   ///< ::urAdapterGet shall only retrieve that number of
                   ///< platforms.
    uint32_t *NumAdapters ///< [out][optional] returns the total number of
                          ///< adapters available.
) {
  if (NumEntries > 0 && Adapters) {
    if (GlobalAdapter) {
      std::lock_guard<std::mutex> Lock{GlobalAdapter->Mutex};
      if (GlobalAdapter->RefCount++ == 0) {
        adapterStateInit();
      }
    } else {
      // If the GetAdapter is called after the Library began or was torndown,
      // then temporarily create a new Adapter handle and register a new
      // cleanup.
      GlobalAdapter = new ur_adapter_handle_t_();
      std::lock_guard<std::mutex> Lock{GlobalAdapter->Mutex};
      if (GlobalAdapter->RefCount++ == 0) {
        adapterStateInit();
      }
      std::atexit(globalAdapterOnDemandCleanup);
    }
    *Adapters = GlobalAdapter;
  }

  if (NumAdapters) {
    *NumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterRelease(ur_adapter_handle_t) {
  // Check first if the Adapter pointer is valid
  if (GlobalAdapter) {
    std::lock_guard<std::mutex> Lock{GlobalAdapter->Mutex};
    if (--GlobalAdapter->RefCount == 0) {
      return adapterStateTeardown();
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterRetain(ur_adapter_handle_t) {
  if (GlobalAdapter) {
    std::lock_guard<std::mutex> Lock{GlobalAdapter->Mutex};
    GlobalAdapter->RefCount++;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urAdapterGetLastError(
    ur_adapter_handle_t,  ///< [in] handle of the platform instance
    const char **Message, ///< [out] pointer to a C string where the adapter
                          ///< specific error message will be stored.
    int32_t *Error ///< [out] pointer to an integer where the adapter specific
                   ///< error code will be stored.
) {
  *Message = ErrorMessage;
  *Error = ErrorAdapterNativeCode;

  return ErrorMessageCode;
}

ur_result_t urAdapterGetInfo(ur_adapter_handle_t, ur_adapter_info_t PropName,
                             size_t PropSize, void *PropValue,
                             size_t *PropSizeRet) {
  UrReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case UR_ADAPTER_INFO_BACKEND:
    return ReturnValue(UR_ADAPTER_BACKEND_LEVEL_ZERO);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(GlobalAdapter->RefCount.load());
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}
} // namespace ur::level_zero
