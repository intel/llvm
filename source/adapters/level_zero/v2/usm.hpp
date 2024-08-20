//===--------- usm.cpp - Level Zero Adapter ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ur_api.h"

#include "common.hpp"
#include "ur_pool_manager.hpp"

struct ur_usm_pool_handle_t_ : _ur_object {
  ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                        ur_usm_pool_desc_t *pPoolDes);

  ur_context_handle_t getContextHandle() const;

private:
  ur_context_handle_t hContext;
};
