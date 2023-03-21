// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#ifndef UR_BACKTRACE_H
#define UR_BACKTRACE_H 1

#include "ur_validation_layer.hpp"

#define MAX_BACKTRACE_FRAMES 64

namespace validation_layer {

using BacktraceLine = std::string;
std::vector<BacktraceLine> getCurrentBacktrace();

} // namespace validation_layer

#endif /* UR_BACKTRACE_H */
