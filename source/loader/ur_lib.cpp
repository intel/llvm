/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_lib.cpp
 *
 */
#include "ur_lib.h"
#include "ur_loader.h"

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////
context_t *context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t(){};

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t(){};

//////////////////////////////////////////////////////////////////////////
__urdlllocal ur_result_t context_t::Init(ur_device_init_flags_t device_flags) {
    ur_result_t result;
    result = loader::context->init();

    if (UR_RESULT_SUCCESS == result) {
        result = urInit();
    }

    return result;
}

} // namespace ur_lib
