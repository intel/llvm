/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UR_LOADER_HPP
#define UR_LOADER_HPP 1

#include "ur_adapter_registry.hpp"
#include "ur_ldrddi.hpp"
#include "ur_lib_loader.hpp"

namespace loader {

struct platform_t {
    platform_t(std::unique_ptr<HMODULE, LibLoader::lib_dtor> handle)
        : handle(std::move(handle)) {}

    std::unique_ptr<HMODULE, LibLoader::lib_dtor> handle;
    ur_result_t initStatus = UR_RESULT_SUCCESS;
    dditable_t dditable = {};
};

using platform_vector_t = std::vector<platform_t>;

class context_t {
  public:
    ur_api_version_t version = UR_API_VERSION_0_6;

    platform_vector_t platforms;
    AdapterRegistry adapter_registry;

    bool forceIntercept = false;

    ur_result_t init();
    bool intercept_enabled = false;
};

extern context_t *context;
extern ur_event_factory_t ur_event_factory;

} // namespace loader

#endif /* UR_LOADER_HPP */
