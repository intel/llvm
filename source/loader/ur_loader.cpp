/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include "ur_loader.hpp"

namespace loader {
///////////////////////////////////////////////////////////////////////////////
context_t *context;

///////////////////////////////////////////////////////////////////////////////
ur_result_t context_t::init() {
    for (const auto &name : adapter_registry) {
        auto handle = LibLoader::loadAdapterLibrary(name.c_str());
        if (handle) {
            platforms.emplace_back(std::move(handle));
        }
    }

    if (platforms.size() == 0) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    forceIntercept = getenv_tobool("UR_ENABLE_LOADER_INTERCEPT");

    if (forceIntercept || platforms.size() > 1) {
        intercept_enabled = true;
    }

    return UR_RESULT_SUCCESS;
}

} // namespace loader
