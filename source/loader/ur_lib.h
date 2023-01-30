/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_lib.h
 *
 */

#ifndef UR_LOADER_LIB_H
#define UR_LOADER_LIB_H 1

#include "ur_api.h"
#include "ur_ddi.h"
#include "ur_util.h"
#include <mutex>
#include <vector>

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////
class context_t {
  public:
#ifdef DYNAMIC_LOAD_LOADER
    HMODULE loader = nullptr;
#endif

    context_t();
    ~context_t();

    std::once_flag initOnce;

    ur_result_t Init(ur_platform_init_flags_t pflags,
                     ur_device_init_flags_t dflags);

    ur_result_t urInit();
    ur_dditable_t urDdiTable = {};
};

extern context_t *context;

} // namespace ur_lib

#endif /* UR_LOADER_LIB_H */
