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

    // Save a copy in the cache for future uses.
    platforms.push_back(std::move(platform));
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t adapterStateInit() { return UR_RESULT_SUCCESS; }

ur_adapter_handle_t_::ur_adapter_handle_t_() {

  Adapter.PlatformCache.Compute = [](Result<PlatformVec> &result) {
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
    if (Adapter.ZeResult == std::nullopt) {
      // Setting these environment variables before running zeInit will enable
      // the validation layer in the Level Zero loader.
      if (UrL0Debug & UR_L0_DEBUG_VALIDATION) {
        setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
        setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
      }

      if (getenv("SYCL_ENABLE_PCI") != nullptr) {
        urPrint(
            "WARNING: SYCL_ENABLE_PCI is deprecated and no longer needed.\n");
      }

      // TODO: We can still safely recover if something goes wrong during the
      // init. Implement handling segfault using sigaction.

      // We must only initialize the driver once, even if urPlatformGet() is
      // called multiple times.  Declaring the return value as "static" ensures
      // it's only called once.
      Adapter.ZeResult = ZE_CALL_NOCHECK(zeInit, (ZE_INIT_FLAG_GPU_ONLY));
    }
    assert(Adapter.ZeResult !=
           std::nullopt); // verify that level-zero is initialized
    PlatformVec platforms;

    // Absorb the ZE_RESULT_ERROR_UNINITIALIZED and just return 0 Platforms.
    if (*Adapter.ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
      result = std::move(platforms);
      return;
    }
    if (*Adapter.ZeResult != ZE_RESULT_SUCCESS) {
      urPrint("zeInit: Level Zero initialization failure\n");
      result = ze2urResult(*Adapter.ZeResult);
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

ur_adapter_handle_t_ Adapter{};

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

    fprintf(stderr, "Check balance of create/destroy calls\n");
    fprintf(stderr,
            "----------------------------------------------------------\n");
    for (const auto &Row : CreateDestroySet) {
      int diff = 0;
      for (auto I = Row.begin(); I != Row.end();) {
        const char *ZeName = (*I).c_str();
        const auto &ZeCount = (*ZeCallCount)[*I];

        bool First = (I == Row.begin());
        bool Last = (++I == Row.end());

        if (Last) {
          fprintf(stderr, " \\--->");
          diff -= ZeCount;
        } else {
          diff += ZeCount;
          if (!First) {
            fprintf(stderr, " | \n");
          }
        }

        fprintf(stderr, "%30s = %-5d", ZeName, ZeCount);
      }

      if (diff) {
        LeakFound = true;
        fprintf(stderr, " ---> LEAK = %d", diff);
      }
      fprintf(stderr, "\n");
    }

    ZeCallCount->clear();
    delete ZeCallCount;
    ZeCallCount = nullptr;
  }
  if (LeakFound)
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;

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
    std::lock_guard<std::mutex> Lock{Adapter.Mutex};
    if (Adapter.RefCount++ == 0) {
      adapterStateInit();
    }
    *Adapters = &Adapter;
  }

  if (NumAdapters) {
    *NumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  std::lock_guard<std::mutex> Lock{Adapter.Mutex};
  if (--Adapter.RefCount == 0) {
    return adapterStateTeardown();
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  std::lock_guard<std::mutex> Lock{Adapter.Mutex};
  Adapter.RefCount++;

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
    return ReturnValue(Adapter.RefCount.load());
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}
