//===----------- common.hpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ur/ur.hpp"
#include "ur2offload.hpp"
#include <atomic>

namespace ur::offload {
struct ddi_getter {
  const static ur_dditable_t *value();
};
using handle_base = ur::handle_base<ur::offload::ddi_getter>;
} // namespace ur::offload

struct RefCounted : ur::offload::handle_base {
  std::atomic_uint32_t RefCount = 1;
};

#define OL_RETURN_ON_ERR(call)                                                 \
  if (auto OlRes = call) {                                                     \
    return offloadResultToUR(OlRes);                                           \
  }
