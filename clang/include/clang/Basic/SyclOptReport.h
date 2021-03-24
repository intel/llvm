//===---------------------- SyclOptReport.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines clang::SyclOptReport class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SYCLOPTREPORT_H
#define LLVM_CLANG_BASIC_SYCLOPTREPORT_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class FunctionDecl;

class SyclOptReport {
  struct OptReportInfo {
    StringRef KernelName;
    StringRef KernelArg;
    StringRef ArgType;
    SourceLocation KernelArgLoc;

    OptReportInfo(StringRef KernelName, StringRef KernelArg, StringRef ArgType,
                  SourceLocation KernelArgLoc)
        : KernelName(KernelName), KernelArg(KernelArg), ArgType(ArgType),
          KernelArgLoc(KernelArgLoc) {}
  };
  llvm::DenseMap<const FunctionDecl *, SmallVector<OptReportInfo, 4>> Map;

public:
  void AddKernelArg(const FunctionDecl *FD, StringRef KernelName,
                    StringRef KernelArg, StringRef ArgType,
                    SourceLocation KernelArgLoc) {
    Map[FD].emplace_back(KernelName, KernelArg, ArgType, KernelArgLoc);
  }

  SmallVector<OptReportInfo, 4> &getInfo(const FunctionDecl *FD) {
    auto It = Map.find(FD);
    return It->second;
  }
};

} // namespace clang

#endif
