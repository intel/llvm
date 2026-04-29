// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_BACKTRACE_H
#define UR_BACKTRACE_H 1

#include <string>
#include <vector>

#define MAX_BACKTRACE_FRAMES 64

namespace ur {

using BacktraceLine = std::string;
std::vector<BacktraceLine> getCurrentBacktrace();

} // namespace ur

#endif /* UR_BACKTRACE_H */
