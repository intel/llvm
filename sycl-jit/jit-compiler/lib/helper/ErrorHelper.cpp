//===- ErrorHelper.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ErrorHelper.h"

#include <sstream>

std::string jit_compiler::formatError(llvm::Error &&Err,
                                      const std::string &Msg) {
  std::stringstream ErrMsg;
  ErrMsg << Msg << "\nDetailed information:\n";
  llvm::handleAllErrors(std::move(Err),
                        [&ErrMsg](const llvm::StringError &StrErr) {
                          ErrMsg << "\t" << StrErr.getMessage() << "\n";
                        });
  return ErrMsg.str();
}
