/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "ur_loader.hpp"

namespace ur_loader {
///////////////////////////////////////////////////////////////////////////////
context_t *getContext() { return context_t::get_direct(); }

///////////////////////////////////////////////////////////////////////////////
ur_result_t context_t::init() {
    for (const auto &adapterPaths : adapter_registry) {
        for (const auto &path : adapterPaths) {
            auto handle = LibLoader::loadAdapterLibrary(path.string().c_str());
            if (handle) {
                platforms.emplace_back(std::move(handle));
                break;
            }
        }
    }

    forceIntercept = getenv_tobool("UR_ENABLE_LOADER_INTERCEPT");

    if (forceIntercept || platforms.size() > 1) {
        intercept_enabled = true;
    }

    return UR_RESULT_SUCCESS;
}

} // namespace ur_loader
