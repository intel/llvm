/*
 *
 * Copyright (C) 2019-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_null.hpp
 *
 */
#ifndef UR_ADAPTER_NULL_H
#define UR_ADAPTER_NULL_H 1

#include "ur_ddi.h"
#include "ur_util.hpp"
#include <stdlib.h>
#include <vector>

namespace driver {
///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t {
  public:
    ur_api_version_t version = UR_API_VERSION_CURRENT;

    ur_dditable_t urDdiTable = {};
    context_t();
    ~context_t() = default;

    void *get() {
        static uint64_t count = 0x80800000;
        return reinterpret_cast<void *>(++count);
    }
};

extern context_t d_context;
} // namespace driver

#endif /* UR_ADAPTER_NULL_H */
