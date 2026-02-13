//==-- SYCLKernelParamOptInfo.cpp -- get kernel param optimization info ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLKernelParamOptInfo.h"

#include "llvm/IR/Constants.h"
#include "llvm/Support/Casting.h"

namespace {

// Must match the one produced by DeadArgumentElimination
static constexpr char MetaDataID[] = "sycl_kernel_omit_args";

} // anonymous namespace

namespace llvm {

void SYCLKernelParamOptInfo::releaseMemory() { clear(); }

SYCLKernelParamOptInfo
SYCLKernelParamOptInfoAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  SYCLKernelParamOptInfo Res;

  for (const Function &F : M) {
    MDNode *MD = F.getMetadata(MetaDataID);
    if (!MD)
      continue;
    using BaseTy = SYCLKernelParamOptInfoBaseTy;
    auto Ins =
        Res.insert(BaseTy::value_type{F.getName(), BaseTy::mapped_type{}});
    assert(Ins.second && "duplicate kernel?");
    BitVector &ParamDropped = Ins.first->second;

    for (const MDOperand &MDOp : MD->operands()) {
      const auto *MDConst = cast<ConstantAsMetadata>(MDOp);
      unsigned ID = static_cast<unsigned>(
          cast<ConstantInt>(MDConst->getValue())->getValue().getZExtValue());
      ParamDropped.push_back(ID != 0);
    }
  }
  return Res;
}

AnalysisKey SYCLKernelParamOptInfoAnalysis::Key;

} // namespace llvm
