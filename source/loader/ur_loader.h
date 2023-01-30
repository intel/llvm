/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UR_LOADER_H
#define UR_LOADER_H 1

#include <vector>

#include "ur_ddi.h"

#include "ur_object.h"

#include "ur_ldrddi.h"

namespace loader {
//////////////////////////////////////////////////////////////////////////
struct platform_t {
    HMODULE handle = NULL;
    ur_result_t initStatus = UR_RESULT_SUCCESS;
    dditable_t dditable = {};
};

using platform_vector_t = std::vector<platform_t>;

///////////////////////////////////////////////////////////////////////////////
class context_t {
  public:
    ur_api_version_t version = UR_API_VERSION_0_9;

    platform_vector_t platforms;

    bool forceIntercept = false;

    ur_result_t init();
    ~context_t();
    bool intercept_enabled = false;
};

extern context_t *context;
extern ur_event_factory_t ur_event_factory;

} // namespace loader

#endif /* UR_LOADER_H */
