/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_lib.cpp
 *
 */
#include "ur_lib.hpp"
#include "logger/ur_logger.hpp"
#include "ur_loader.hpp"

#include <cstring>

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////
context_t *context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() {
    for (auto l : layers) {
        if (l->isAvailable()) {
            for (auto &layerName : l->getNames()) {
                availableLayers += layerName + ";";
            }
        }
    }
    // Remove the trailing ";"
    availableLayers.pop_back();
    parseEnvEnabledLayers();
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

bool context_t::layerExists(const std::string &layerName) const {
    return availableLayers.find(layerName) != std::string::npos;
}

void context_t::parseEnvEnabledLayers() {
    auto maybeEnableEnvVarMap = getenv_to_map("UR_ENABLE_LAYERS", false);
    if (!maybeEnableEnvVarMap.has_value()) {
        return;
    }
    auto enableEnvVarMap = maybeEnableEnvVarMap.value();

    for (auto &key : enableEnvVarMap) {
        enabledLayerNames.insert(key.first);
    }
}

void context_t::initLayers() const {
    for (auto &l : layers) {
        if (l->isAvailable()) {
            l->init(&context->urDdiTable, enabledLayerNames, codelocData);
        }
    }
}

void context_t::tearDownLayers() const {
    for (auto &l : layers) {
        if (l->isAvailable()) {
            l->tearDown();
        }
    }
}

//////////////////////////////////////////////////////////////////////////
__urdlllocal ur_result_t context_t::Init(
    ur_device_init_flags_t, ur_loader_config_handle_t hLoaderConfig) {
    ur_result_t result;
    const char *logger_name = "loader";
    logger::init(logger_name);
    logger::debug("Logger {} initialized successfully!", logger_name);

    result = ur_loader::context->init();

    if (UR_RESULT_SUCCESS == result) {
        result = urLoaderInit();
    }

    if (hLoaderConfig) {
        codelocData = hLoaderConfig->codelocData;
        enabledLayerNames.merge(hLoaderConfig->getEnabledLayerNames());
    }

    if (!enabledLayerNames.empty()) {
        initLayers();
    }

    return result;
}

ur_result_t urLoaderConfigCreate(ur_loader_config_handle_t *phLoaderConfig) {
    if (!phLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    *phLoaderConfig = new ur_loader_config_handle_t_;
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigRetain(ur_loader_config_handle_t hLoaderConfig) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    hLoaderConfig->incrementReferenceCount();
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigRelease(ur_loader_config_handle_t hLoaderConfig) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    if (hLoaderConfig->decrementReferenceCount() == 0) {
        delete hLoaderConfig;
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigGetInfo(ur_loader_config_handle_t hLoaderConfig,
                                  ur_loader_config_info_t propName,
                                  size_t propSize, void *pPropValue,
                                  size_t *pPropSizeRet) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }

    if (!pPropValue && !pPropSizeRet) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    switch (propName) {
    case UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS: {
        if (pPropSizeRet) {
            *pPropSizeRet = context->availableLayers.size() + 1;
        }
        if (pPropValue) {
            char *outString = static_cast<char *>(pPropValue);
            if (propSize != context->availableLayers.size() + 1) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            std::memcpy(outString, context->availableLayers.data(),
                        propSize - 1);
            outString[propSize - 1] = '\0';
        }
        break;
    }
    case UR_LOADER_CONFIG_INFO_REFERENCE_COUNT: {
        auto refCount = hLoaderConfig->getReferenceCount();
        auto truePropSize = sizeof(refCount);
        if (pPropSizeRet) {
            *pPropSizeRet = truePropSize;
        }
        if (pPropValue) {
            if (propSize != truePropSize) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            std::memcpy(pPropValue, &refCount, truePropSize);
        }
        break;
    }
    default:
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigEnableLayer(ur_loader_config_handle_t hLoaderConfig,
                                      const char *pLayerName) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    if (!pLayerName) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    if (!context->layerExists(std::string(pLayerName))) {
        return UR_RESULT_ERROR_LAYER_NOT_PRESENT;
    }
    hLoaderConfig->enabledLayers.insert(pLayerName);
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderTearDown() {
    context->tearDownLayers();

    return UR_RESULT_SUCCESS;
}

ur_result_t
urLoaderConfigSetCodeLocationCallback(ur_loader_config_handle_t hLoaderConfig,
                                      ur_code_location_callback_t pfnCodeloc,
                                      void *pUserData) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    if (!pfnCodeloc) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    hLoaderConfig->codelocData.codelocCb = pfnCodeloc;
    hLoaderConfig->codelocData.codelocUserdata = pUserData;

    return UR_RESULT_SUCCESS;
}

} // namespace ur_lib
