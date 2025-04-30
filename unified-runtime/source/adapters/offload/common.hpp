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

#include <atomic>

namespace ur::offload {
struct handle_base {};
} // namespace ur::offload

struct RefCounted : ur::offload::handle_base {
  std::atomic_uint32_t RefCount = 1;
};
