// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_BACKTRACE_H
#define UR_BACKTRACE_H 1

#include <string>
#include <vector>

namespace ur_validation_layer {

using BacktraceLine = std::string;
std::vector<BacktraceLine> getCurrentBacktrace();

} // namespace ur_validation_layer

#endif /* UR_BACKTRACE_H */
