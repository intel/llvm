//===--------------- AddAspectNames.cpp - AddAspectNames Pass
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//
//

#include "llvm/SYCLLowerIR/AddAspectNames.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses AddAspectNamesPass::run(Module &M,
                                          ModuleAnalysisManager &MAM) {
  SmallDenseMap<int64_t, Metadata *, 128> ValueToNameValuePairMD;
  if (NamedMDNode *Node = M.getNamedMetadata("sycl_aspects")) {
    for (MDNode *N : Node->operands()) {
      assert(N->getNumOperands() == 2 &&
             "Each operand of sycl_aspects must be a pair.");

      // The aspect's integral value is the second operand.
      const auto *AspectCAM = cast<ConstantAsMetadata>(N->getOperand(1));
      const Constant *AspectC = AspectCAM->getValue();

      ValueToNameValuePairMD[cast<ConstantInt>(AspectC)->getSExtValue()] = N;
    }
  }

  auto &Ctx = M.getContext();
  for (Function &F : M.functions()) {
    auto *MDNode = F.getMetadata("sycl_used_aspects");
    if (!MDNode)
      continue;

    // Change the metadata from {1, 2} to
    // a format like {{"cpu", 1}, {"gpu", 2}}
    SmallVector<Metadata *, 8> AspectNameValuePairs;
    for (const auto &MDOp : MDNode->operands()) {
      const Constant *C = cast<ConstantAsMetadata>(MDOp)->getValue();
      int64_t AspectValue = cast<ConstantInt>(C)->getSExtValue();
      if (auto it = ValueToNameValuePairMD.find(AspectValue);
          it != ValueToNameValuePairMD.end())
        AspectNameValuePairs.push_back(it->second);
      else
        AspectNameValuePairs.push_back(MDOp);
    }

    F.setMetadata("sycl_used_aspects", MDNode::get(Ctx, AspectNameValuePairs));
  }

  return PreservedAnalyses::all();
}