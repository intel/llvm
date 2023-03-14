/*
 *
 * Copyright (C) 2023 Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_proxy_layer.h
 *
 */
#ifndef UR_PROXY_LAYER_H
#define UR_PROXY_LAYER_H 1

#include "ur_ddi.h"
#include "ur_util.hpp"

///////////////////////////////////////////////////////////////////////////////
class __urdlllocal proxy_layer_context_t {
  public:
    ur_api_version_t version = UR_API_VERSION_0_6;

    virtual bool isEnabled() = 0;
    virtual ur_result_t init(ur_dditable_t *dditable) = 0;
};

#endif /* UR_PROXY_LAYER_H */