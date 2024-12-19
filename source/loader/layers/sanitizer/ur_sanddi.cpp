/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanddi.cpp
 *
 */

#include "asan/asan_ddi.hpp"
#include "msan/msan_ddi.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

ur_result_t context_t::init(ur_dditable_t *dditable,
                            const std::set<std::string> &enabledLayerNames,
                            [[maybe_unused]] codeloc_data codelocData) {
    bool asanEnabled = enabledLayerNames.count("UR_LAYER_ASAN");
    bool msanEnabled = enabledLayerNames.count("UR_LAYER_MSAN");

    if (asanEnabled && msanEnabled) {
        getContext()->logger.warning(
            "Enabling ASAN and MSAN at the same time is not "
            "supported.");
        return UR_RESULT_SUCCESS;
    } else if (asanEnabled) {
        enabledType = SanitizerType::AddressSanitizer;
    } else if (msanEnabled) {
        enabledType = SanitizerType::MemorySanitizer;
    } else {
        return UR_RESULT_SUCCESS;
    }

    urDdiTable = *dditable;

    switch (enabledType) {
    case SanitizerType::AddressSanitizer:
        initAsanInterceptor();
        return initAsanDDITable(dditable);
    case SanitizerType::MemorySanitizer:
        initMsanInterceptor();
        return initMsanDDITable(dditable);
    default:
        break;
    }

    return UR_RESULT_SUCCESS;
}

} // namespace ur_sanitizer_layer
