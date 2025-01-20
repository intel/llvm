/*
 *
 * Copyright (C) 2019-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_null.hpp
 *
 */
#include "ur_api.h"
#ifndef UR_ADAPTER_MOCK_H
#define UR_ADAPTER_MOCK_H 1

#include "ur_ddi.h"
#include "ur_util.hpp"

namespace driver {
///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t {
public:
  ur_api_version_t version = UR_API_VERSION_CURRENT;

  ur_dditable_t urDdiTable = {};
  context_t();
  ~context_t() = default;

  ur_adapter_handle_t adapter = reinterpret_cast<ur_adapter_handle_t>(1);
  ur_device_handle_t device = reinterpret_cast<ur_device_handle_t>(2);
  ur_platform_handle_t platform = reinterpret_cast<ur_platform_handle_t>(3);
};

extern context_t d_context;

} // namespace driver

#endif /* UR_ADAPTER_MOCK_H */
