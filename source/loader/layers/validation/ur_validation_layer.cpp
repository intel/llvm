/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_validation_layer.cpp
 *
 */
#include "ur_validation_layer.hpp"

namespace validation_layer {
context_t context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() : logger(logger::create_logger("validation")) {
    enableValidation = getenv_tobool("UR_ENABLE_VALIDATION_LAYER");
    enableParameterValidation = getenv_tobool("UR_ENABLE_PARAMETER_VALIDATION");
    enableLeakChecking = getenv_tobool("UR_ENABLE_LEAK_CHECKING");
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

} // namespace validation_layer
