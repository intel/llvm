/*
 *
 * Copyright (C) 2023 Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer.h
 *
 */

#ifndef UR_SANITIZER_LAYER_H
#define UR_SANITIZER_LAYER_H 1

#include "ur_ddi.h"
#include "ur_proxy_layer.hpp"
#include "ur_util.hpp"

#define SANITIZER_COMP_NAME "sanitizer layer"

class SanitizerInterceptor;

namespace ur_sanitizer_layer {
///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t : public proxy_layer_context_t {
  public:
    ur_dditable_t urDdiTable = {};
    SanitizerInterceptor *interceptor = {};

    context_t();
    ~context_t();

    bool isAvailable() const override;

    std::vector<std::string> getNames() const override { return {name}; }
    ur_result_t init(ur_dditable_t *dditable,
                     const std::set<std::string> &enabledLayerNames) override;

  private:
    const std::string name = "UR_LAYER_SANITIZER";
};

extern context_t context;
} // namespace ur_sanitizer_layer

#endif /* UR_sanitizer_LAYER_H */
