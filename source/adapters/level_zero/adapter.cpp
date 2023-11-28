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

ur_adapter_handle_t_ Adapter{};

UR_APIEXPORT ur_result_t UR_APICALL
urInit(ur_device_init_flags_t
           DeviceFlags, ///< [in] device initialization flags.
                        ///< must be 0 (default) or a combination of
                        ///< ::ur_device_init_flag_t.
       ur_loader_config_handle_t) {
  std::ignore = DeviceFlags;

  return UR_RESULT_SUCCESS;
}

ur_result_t adapterStateTeardown() {
  // reclaim ur_platform_handle_t objects here since we don't have
  // urPlatformRelease.
  for (ur_platform_handle_t Platform : *URPlatformsCache) {
    delete Platform;
  }
  delete URPlatformsCache;
  delete URPlatformsCacheMutex;

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

UR_APIEXPORT ur_result_t UR_APICALL urTearDown(
    void *Params ///< [in] pointer to tear down parameters
) {
  std::ignore = Params;
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
    // TODO: Some initialization that happens in urPlatformsGet could be moved
    // here for when RefCount reaches 1
    Adapter.RefCount++;
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
    [[maybe_unused]] ur_adapter_handle_t
        AdapterHandle,    ///< [in] handle of the platform instance
    const char **Message, ///< [out] pointer to a C string where the adapter
                          ///< specific error message will be stored.
    [[maybe_unused]] int32_t
        *Error ///< [out] pointer to an integer where the adapter specific
               ///< error code will be stored.
) {
  AdapterHandle = &Adapter;
  *Message = ErrorMessage;
  Error = &ErrorAdapterNativeCode;

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
