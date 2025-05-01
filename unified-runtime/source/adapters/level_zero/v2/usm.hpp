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

struct ur_usm_pool_handle_t_ : ur_object {
  ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                        ur_usm_pool_desc_t *pPoolDes);

  ur_context_handle_t getContextHandle() const;

  ur_result_t allocate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                       const ur_usm_desc_t *pUSMDesc, ur_usm_type_t type,
                       size_t size, void **ppRetMem);
  ur_result_t free(void *ptr);

private:
  ur_context_handle_t hContext;
  usm::pool_manager<usm::pool_descriptor, umf_memory_pool_t> poolManager;

  umf_memory_pool_handle_t getPool(const usm::pool_descriptor &desc);
};
