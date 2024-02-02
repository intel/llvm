// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>
#include <vector>

#define MAX_BACKTRACE_FRAMES 64

namespace ur_sanitizer_layer {

using BacktraceLine = std::string;
std::vector<BacktraceLine> GetCurrentBacktrace();

} // namespace ur_sanitizer_layer
