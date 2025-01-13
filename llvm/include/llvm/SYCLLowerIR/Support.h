//===----- Support.h - A set of utility functions for the tool ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file contains a set of utility functions and macros used through
// the overall code of the tool.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#define AssertRelease(Cond, Msg)                                               \
  do {                                                                         \
    if (!(Cond))                                                               \
      llvm::report_fatal_error(llvm::Twine(__FILE__ " ") + (Msg));             \
  } while (false)

#define CHECK_AND_EXIT(E)                                                      \
  if (Error LocE = E) {                                                        \
    logAllUnhandledErrors(std::move(LocE), WithColor::error(errs()));          \
    exit(1);                                                                   \
  }

namespace llvm {

inline void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

inline void warning(const Twine &Msg) {
  errs() << "sycl-post-link WARNING: " << Msg << '\n';
}

inline void checkError(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

} // namespace llvm
