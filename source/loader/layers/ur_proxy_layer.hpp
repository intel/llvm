/*
 *
 * Copyright (C) 2023 Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file ur_proxy_layer.h
 *
 */
#ifndef UR_PROXY_LAYER_H
#define UR_PROXY_LAYER_H 1

#include "ur_codeloc.hpp"
#include "ur_ddi.h"
#include "ur_util.hpp"

#include <set>

///////////////////////////////////////////////////////////////////////////////
class __urdlllocal proxy_layer_context_t {
public:
  virtual ~proxy_layer_context_t() {};
  ur_api_version_t version = UR_API_VERSION_CURRENT;

  virtual ur_result_t init(ur_dditable_t *dditable,
                           const std::set<std::string> &enabledLayerNames,
                           codeloc_data codelocData) = 0;
  virtual ur_result_t tearDown() = 0;
};

#endif /* UR_PROXY_LAYER_H */
