//===---------------------- SyclOptReportHandler.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines clang::SyclOptReportHandler class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SYCLOPTREPORTHANDLER_H
#define LLVM_CLANG_BASIC_SYCLOPTREPORTHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class FunctionDecl;

class SyclOptReportHandler {
private:
  struct OptReportInfo {
    std::string KernelArgName;
    std::string KernelArgType;
    SourceLocation KernelArgLoc;

    OptReportInfo(std::string ArgName, std::string ArgType,
                  SourceLocation ArgLoc)
        : KernelArgName(std::move(ArgName)), KernelArgType(std::move(ArgType)),
          KernelArgLoc(ArgLoc) {}
  };
  llvm::DenseMap<const FunctionDecl *, SmallVector<OptReportInfo>> Map;

public:
  void AddKernelArgs(const FunctionDecl *FD, std::string ArgName,
                     std::string ArgType, SourceLocation ArgLoc) {
    Map[FD].emplace_back(ArgName, ArgType, ArgLoc);
  }
  SmallVector<OptReportInfo> &GetInfo(const FunctionDecl *FD) {
    auto It = Map.find(FD);
    assert(It != Map.end());
    return It->second;
  }

  bool HasOptReportInfo(const FunctionDecl *FD) const {
    return Map.find(FD) != Map.end();
  }
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_SYCLOPTREPORTHANDLER_H
