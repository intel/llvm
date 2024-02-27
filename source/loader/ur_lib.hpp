/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_lib.hpp
 *
 */

#ifndef UR_LOADER_LIB_H
#define UR_LOADER_LIB_H 1

#include "ur_api.h"
#include "ur_codeloc.hpp"
#include "ur_ddi.h"
#include "ur_proxy_layer.hpp"
#include "ur_util.hpp"

#include "validation/ur_validation_layer.hpp"
#if UR_ENABLE_TRACING
#include "tracing/ur_tracing_layer.hpp"
#endif
#if UR_ENABLE_SANITIZER
#include "sanitizer/ur_sanitizer_layer.hpp"
#endif

#include <atomic>
#include <mutex>
#include <set>
#include <vector>

struct ur_loader_config_handle_t_ {
    std::set<std::string> enabledLayers;
    std::atomic_uint32_t refCount = 1;

    uint32_t incrementReferenceCount() {
        return refCount.fetch_add(1, std::memory_order_acq_rel) + 1;
    }
    uint32_t decrementReferenceCount() {
        return refCount.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }
    uint32_t getReferenceCount() {
        return refCount.load(std::memory_order_acquire);
    }
    std::set<std::string> &getEnabledLayerNames() { return enabledLayers; }

    codeloc_data codelocData;
};

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t {
  public:
#ifdef DYNAMIC_LOAD_LOADER
    HMODULE loader = nullptr;
#endif

    context_t();
    ~context_t();

    std::once_flag initOnce;

    ur_result_t Init(ur_device_init_flags_t dflags,
                     ur_loader_config_handle_t hLoaderConfig);

    ur_result_t urLoaderInit();
    ur_dditable_t urDdiTable = {};

    const std::vector<proxy_layer_context_t *> layers = {
        &ur_validation_layer::context,
#if UR_ENABLE_TRACING
        &ur_tracing_layer::context,
#endif
#if UR_ENABLE_SANITIZER
        &ur_sanitizer_layer::context
#endif
    };
    std::string availableLayers;
    std::set<std::string> enabledLayerNames;

    codeloc_data codelocData;

    bool layerExists(const std::string &layerName) const;
    void parseEnvEnabledLayers();
    void initLayers() const;
    void tearDownLayers() const;
};

extern context_t *context;
ur_result_t urLoaderConfigCreate(ur_loader_config_handle_t *phLoaderConfig);
ur_result_t urLoaderConfigRetain(ur_loader_config_handle_t hLoaderConfig);
ur_result_t urLoaderConfigRelease(ur_loader_config_handle_t hLoaderConfig);
ur_result_t urLoaderConfigGetInfo(ur_loader_config_handle_t hLoaderConfig,
                                  ur_loader_config_info_t propName,
                                  size_t propSize, void *pPropValue,
                                  size_t *pPropSizeRet);
ur_result_t urLoaderConfigEnableLayer(ur_loader_config_handle_t hLoaderConfig,
                                      const char *pLayerName);
ur_result_t urLoaderTearDown();
ur_result_t
urLoaderConfigSetCodeLocationCallback(ur_loader_config_handle_t hLoaderConfig,
                                      ur_code_location_callback_t pfnCodeloc,
                                      void *pUserData);

ur_result_t urDeviceGetSelected(ur_platform_handle_t hPlatform,
                                ur_device_type_t DeviceType,
                                uint32_t NumEntries,
                                ur_device_handle_t *phDevices,
                                uint32_t *pNumDevices);
} // namespace ur_lib
#endif /* UR_LOADER_LIB_H */
