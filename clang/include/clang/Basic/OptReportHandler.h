//===------------------------ OptReportHandler.h ----------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_BASIC_OPTREPORTHANDLER_H
#define LLVM_CLANG_BASIC_OPTREPORTHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class FunctionDecl;

class SyclOptReportHandler {
private:
  struct OptReportInfo {
    std::string KernelArgDescName; // Kernel argument name itself, or the name
                                   // of the parent class if the kernel argument
                                   // is a decomposed member.
    std::string KernelArgType;
    SourceLocation KernelArgLoc;
    unsigned KernelArgSize;
    std::string KernelArgDesc;
    std::string KernelArgDecomposedField;

    OptReportInfo(std::string ArgDescName, std::string ArgType,
                  SourceLocation ArgLoc, unsigned ArgSize, std::string ArgDesc,
                  std::string ArgDecomposedField)
        : KernelArgDescName(std::move(ArgDescName)),
          KernelArgType(std::move(ArgType)), KernelArgLoc(ArgLoc),
          KernelArgSize(ArgSize), KernelArgDesc(std::move(ArgDesc)),
          KernelArgDecomposedField(std::move(ArgDecomposedField)) {}
  };
  llvm::DenseMap<const FunctionDecl *, SmallVector<OptReportInfo>> Map;

public:
  void AddKernelArgs(const FunctionDecl *FD, StringRef ArgDescName,
                     StringRef ArgType, SourceLocation ArgLoc, unsigned ArgSize,
                     StringRef ArgDesc, StringRef ArgDecomposedField) {
    Map[FD].emplace_back(ArgDescName.data(), ArgType.data(), ArgLoc, ArgSize,
                         ArgDesc.data(), ArgDecomposedField.data());
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

#endif // LLVM_CLANG_BASIC_OPTREPORTHANDLER_H
