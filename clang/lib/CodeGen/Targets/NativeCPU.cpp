//===- NativeCPU.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

namespace {
class NativeCPUABIInfo : public DefaultABIInfo {
private:
  const ABIInfo *HostABIInfo;

public:
  NativeCPUABIInfo(CodeGen::CodeGenTypes &CGT, const ABIInfo *HostABIInfo)
      : DefaultABIInfo(CGT), HostABIInfo(HostABIInfo) {}
};

class NativeCPUTargetCodeGenInfo : public TargetCodeGenInfo {
private:
  std::unique_ptr<TargetCodeGenInfo> HostTargetCodeGenInfo;

public:
  NativeCPUTargetCodeGenInfo(
      CodeGen::CodeGenTypes &CGT,
      std::unique_ptr<TargetCodeGenInfo> HostTargetCodeGenInfo)
      : TargetCodeGenInfo(std::make_unique<NativeCPUABIInfo>(
            CGT, HostTargetCodeGenInfo ? &HostTargetCodeGenInfo->getABIInfo()
                                       : nullptr)),
        HostTargetCodeGenInfo(std::move(HostTargetCodeGenInfo)) {}
};
} // namespace

std::unique_ptr<TargetCodeGenInfo> CodeGen::createNativeCPUTargetCodeGenInfo(
    CodeGenModule &CGM,
    std::unique_ptr<TargetCodeGenInfo> HostTargetCodeGenInfo) {
  return std::make_unique<NativeCPUTargetCodeGenInfo>(
      CGM.getTypes(), std::move(HostTargetCodeGenInfo));
}
