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
context_t::context_t() : logger(logger::create_logger("validation")) {
    enableValidation = getenv_tobool("UR_ENABLE_VALIDATION_LAYER");
    enableParameterValidation = getenv_tobool("UR_ENABLE_PARAMETER_VALIDATION");
    enableLeakChecking = getenv_tobool("UR_ENABLE_LEAK_CHECKING");
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

} // namespace ur_validation_layer
