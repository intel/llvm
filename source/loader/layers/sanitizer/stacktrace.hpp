/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file stacktrace.hpp
 *
 */

#pragma once

#include "common.hpp"

#include <string>
#include <vector>

namespace ur_sanitizer_layer {

#define MAX_BACKTRACE_FRAMES 64

struct StackTrace {
    std::vector<AddressInfo> stack;

    void Print();
};

StackTrace GetCurrentBacktrace();

} // namespace ur_sanitizer_layer