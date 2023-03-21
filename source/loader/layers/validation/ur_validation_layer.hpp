/*
 *
 * Copyright (C) 2023 Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_layer.h
 *
 */
#pragma once
#include "logger/ur_logger.hpp"
#include "ur_ddi.h"
#include "ur_proxy_layer.hpp"
#include "ur_util.hpp"

namespace validation_layer {

///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t : public proxy_layer_context_t {
  public:
    bool enableValidation = false;
    bool enableParameterValidation = false;
    bool enableLeakChecking = false;

    logger::Logger logger;

    ur_dditable_t urDdiTable = {};

    context_t();
    ~context_t();

    bool isEnabled() { return enableValidation; };
    ur_result_t init(ur_dditable_t *dditable);
};

extern context_t context;

} // namespace validation_layer
