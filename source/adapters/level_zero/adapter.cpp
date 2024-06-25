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

// Due to multiple DLLMain definitions with SYCL, Global Adapter is init at
// variable creation.
#if defined(_WIN32)
ur_adapter_handle_t_ *GlobalAdapter = new ur_adapter_handle_t_();
#else
ur_adapter_handle_t_ *GlobalAdapter;
#endif

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

  ~ur_legacy_sink() = default;
};

ur_result_t initPlatforms(PlatformVec &platforms) noexcept try {
  uint32_t ZeDriverCount = 0;
  ZE2UR_CALL(zeDriverGet, (&ZeDriverCount, nullptr));
  if (ZeDriverCount == 0) {
    return UR_RESULT_SUCCESS;
  }

  std::vector<ze_driver_handle_t> ZeDrivers;
  ZeDrivers.resize(ZeDriverCount);

  ZE2UR_CALL(zeDriverGet, (&ZeDriverCount, ZeDrivers.data()));
  for (uint32_t I = 0; I < ZeDriverCount; ++I) {
    auto platform = std::make_unique<ur_platform_handle_t_>(ZeDrivers[I]);
    UR_CALL(platform->initialize());
    ZE2UR_CALL(zelLoaderTranslateHandle,
               (ZEL_HANDLE_DRIVER, platform->ZeDriver,
                (void **)&platform->ZeDriverHandleExpTranslated));

    // Save a copy in the cache for future uses.
    platforms.push_back(std::move(platform));
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t adapterStateInit() { return UR_RESULT_SUCCESS; }

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
      GlobalAdapter->ZeResult =
          ZE_CALL_NOCHECK(zeInit, (ZE_INIT_FLAG_GPU_ONLY));
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

    ur_result_t err = initPlatforms(platforms);
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
  bool LeakFound = false;

  // Print the balance of various create/destroy native calls.
  // The idea is to verify if the number of create(+) and destroy(-) calls are
  // matched.
  if (ZeCallCount && (UrL0LeaksDebug) != 0) {
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
  }
  if (LeakFound)
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
    // Due to multiple DLLMain definitions with SYCL, register to cleanup the
    // Global Adapter after refcnt is 0
#if defined(_WIN32)
  std::atexit(globalAdapterOnDemandCleanup);
#endif

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGet(
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

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  // Check first if the Adapter pointer is valid
  if (GlobalAdapter) {
    std::lock_guard<std::mutex> Lock{GlobalAdapter->Mutex};
    if (--GlobalAdapter->RefCount == 0) {
      return adapterStateTeardown();
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  if (GlobalAdapter) {
    std::lock_guard<std::mutex> Lock{GlobalAdapter->Mutex};
    GlobalAdapter->RefCount++;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetLastError(
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

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetInfo(ur_adapter_handle_t,
                                                     ur_adapter_info_t PropName,
                                                     size_t PropSize,
                                                     void *PropValue,
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

  return UR_RESULT_SUCCESS;
}
