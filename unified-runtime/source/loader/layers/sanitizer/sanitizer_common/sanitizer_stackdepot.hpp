/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_stackdepot.hpp
 *
 */

#pragma once

#include "sanitizer_stacktrace.hpp"

namespace ur_sanitizer_layer {

uint32_t StackDepotPut(StackTrace &Stack);
StackTrace StackDepotGet(uint32_t Id);

} // namespace ur_sanitizer_layer
