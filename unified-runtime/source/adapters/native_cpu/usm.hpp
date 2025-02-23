//===--------- usm.hpp - Native CPU Adapter --------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

namespace native_cpu {
namespace detail {
uint64_t maxUSMAllocationSize(const ur_device_handle_t &Device);
} // namespace detail
} // namespace native_cpu
