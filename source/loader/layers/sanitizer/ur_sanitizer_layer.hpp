/*
 *
 * Copyright (C) 2023 Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer.hpp
 *
 */

#pragma once

#include "logger/ur_logger.hpp"
#include "ur_proxy_layer.hpp"

#define SANITIZER_COMP_NAME "sanitizer layer"

namespace ur_sanitizer_layer {

class SanitizerInterceptor;

enum class SanitizerType {
    None,
    AddressSanitizer,
    MemorySanitizer,
    ThreadSanitizer,
};

///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t : public proxy_layer_context_t {
  public:
    ur_dditable_t urDdiTable = {};
    std::unique_ptr<SanitizerInterceptor> interceptor;
    logger::Logger logger;
    SanitizerType enabledType = SanitizerType::None;

    context_t();
    ~context_t();

    bool isAvailable() const override;

    std::vector<std::string> getNames() const override {
        return {"UR_LAYER_ASAN", "UR_LAYER_MSAN", "UR_LAYER_TSAN"};
    }
    ur_result_t init(ur_dditable_t *dditable,
                     const std::set<std::string> &enabledLayerNames,
                     codeloc_data codelocData) override;

    ur_result_t tearDown() override;
};

extern context_t context;
} // namespace ur_sanitizer_layer
