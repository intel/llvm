//===----------- Utils.h - utilities for sycl-post-link tool --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functions and variables that are used commonly.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

inline void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

inline void checkError(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

} // namespace llvm
