/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_validation_layer.cpp
 *
 */
#include "ur_validation_layer.hpp"

namespace ur_validation_layer {
context_t context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() : logger(logger::create_logger("validation")) {}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

bool context_t::isEnabled(const std::set<std::string> &enabledLayerNames) {
    if (enabledLayerNames.find(nameFullValidation) != enabledLayerNames.end()) {
        enableParameterValidation = true;
        enableLeakChecking = true;
    } else {
        if (enabledLayerNames.find(nameParameterValidation) !=
            enabledLayerNames.end()) {
            enableParameterValidation = true;
        }
        if (enabledLayerNames.find(nameLeakChecking) !=
            enabledLayerNames.end()) {
            enableLeakChecking = true;
        }
    }
    return enableParameterValidation || enableLeakChecking;
}

} // namespace ur_validation_layer
