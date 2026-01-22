//===--------- graph.hpp - Level Zero Adapter -----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../external/driver_experimental/zex_graph.h"
#include "common.hpp"
#include "context.hpp"
#include "unified-runtime/ur_api.h"

namespace ur::level_zero::v2 {

struct ur_exp_graph_handle_t_ : v2::ur_handle_base_t {
public:
  ur_exp_graph_handle_t_(ur_context_handle_t hContext);
  ur_exp_graph_handle_t_(ur_context_handle_t hContext,
                         ze_graph_handle_t zeGraph);
  ~ur_exp_graph_handle_t_();

  ur_exp_graph_handle_t_(const ur_exp_graph_handle_t_ &) = delete;
  ur_exp_graph_handle_t_ &operator=(const ur_exp_graph_handle_t_ &) = delete;

  ze_graph_handle_t getZeHandle() { return zeGraph; }
  ur_context_handle_t getContext() { return hContext; }

private:
  ur_context_handle_t hContext = nullptr;
  ze_graph_handle_t zeGraph = nullptr;
};

struct ur_exp_executable_graph_handle_t_ : v2::ur_handle_base_t {
public:
  ur_exp_executable_graph_handle_t_(ur_context_handle_t hContext,
                                    ur_exp_graph_handle_t hGraph);
  ~ur_exp_executable_graph_handle_t_();

  ur_exp_executable_graph_handle_t_(const ur_exp_executable_graph_handle_t_ &) =
      delete;
  ur_exp_executable_graph_handle_t_ &
  operator=(const ur_exp_executable_graph_handle_t_ &) = delete;

  ze_executable_graph_handle_t &getZeHandle() { return zeExGraph; }
  ur_context_handle_t getContext() const { return hContext; }

private:
  ur_context_handle_t hContext = nullptr;
  ze_executable_graph_handle_t zeExGraph = nullptr;
};

inline bool checkGraphExtensionSupport(ur_context_handle_t hContext) {
  return v2::v2_cast(hContext)->getPlatform()->ZeGraphExt.Supported;
}

} // namespace ur::level_zero::v2
