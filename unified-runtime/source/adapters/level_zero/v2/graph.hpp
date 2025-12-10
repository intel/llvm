//===--------- graph.hpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../external/driver_experimental/zex_graph.h"
#include "common.hpp"
#include "context.hpp"
#include "ur_api.h"

struct ur_exp_graph_handle_t_ : ur_object {
public:
  ur_exp_graph_handle_t_(ur_context_handle_t hContext);
  ur_exp_graph_handle_t_(ur_context_handle_t hContext,
                         ze_graph_handle_t zeGraph);
  ~ur_exp_graph_handle_t_();

  ze_graph_handle_t getZeHandle() { return zeGraph; }
  ur_context_handle_t getContext() { return hContext; }

private:
  ur_context_handle_t hContext = nullptr;
  ze_graph_handle_t zeGraph = nullptr;
};

struct ur_exp_executable_graph_handle_t_ : ur_object {
public:
  ur_exp_executable_graph_handle_t_(ur_context_handle_t hContext,
                                    ur_exp_graph_handle_t hGraph);
  ~ur_exp_executable_graph_handle_t_();

  ze_executable_graph_handle_t &getZeHandle() { return zeExGraph; }
  ur_context_handle_t getContext() const { return hContext; }

private:
  ur_context_handle_t hContext = nullptr;
  ze_executable_graph_handle_t zeExGraph = nullptr;
};

inline bool checkGraphExtensionSupport(ur_context_handle_t hContext) {
  return hContext->getPlatform()->ZeGraphExt.Supported;
}
