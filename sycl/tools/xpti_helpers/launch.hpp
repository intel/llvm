//==----------------- launch.hpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <string>
#include <unistd.h>
#include <vector>

inline std::vector<const char *> toCStyle(const std::vector<std::string> &Arr) {
  std::vector<const char *> CArr;
  CArr.reserve(Arr.size() + 1);
  std::transform(Arr.begin(), Arr.end(), std::back_inserter(CArr),
                 [](const std::string &str) { return str.data(); });

  CArr.push_back(nullptr);

  return CArr;
}

/// Launches external application.
///
/// \param Cmd is a path (full, relative, executable name in PATH) to executable
/// \param Args is program arguments. First argument is executable name. Last
///        argument is nullptr.
/// \param Env is program environment variables. Last variable is nullptr.
int launch(const std::string &Cmd, const std::vector<std::string> &Args,
           const std::vector<std::string> &Env) {
  std::vector<const char *> CArgs = toCStyle(Args);
  std::vector<const char *> CEnv = toCStyle(Env);
  return execve(Cmd.data(), const_cast<char *const *>(CArgs.data()),
                const_cast<char *const *>(CEnv.data()));
}
