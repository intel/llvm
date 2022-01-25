//==----------------- launch.hpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <unistd.h>
#include <vector>

/// Launches external application.
///
/// \param Cmd is a path (full, relative, executable name in PATH) to executable
/// \param Args is program arguments. First argument is executable name. Last
///        argument is nullptr.
/// \param Env is program environment variables. Last variable is nullptr.
int launch(const char *Cmd, const std::vector<const char *> &Args,
           const std::vector<const char *> &Env) {
  return execve(Cmd, const_cast<char *const *>(Args.data()),
                const_cast<char *const *>(Env.data()));
}
