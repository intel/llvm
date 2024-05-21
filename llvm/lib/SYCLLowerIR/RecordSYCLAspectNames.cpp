//===-------- RecordSYCLAspectNames.cpp - RecordSYCLAspectNames Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The !sycl_used_aspects metadata is populated from C++ attributes and
// further populated by the SYCLPropagateAspectsPass that describes which
// apsects a function uses. The format of this metadata initially just an
// integer value corresponding to the enum value in C++. The !sycl_aspects
// named metadata contains the associations from aspect values to aspect names.
// These associations are needed later in sycl-post-link, but we drop
// !sycl_aspects before that to avoid LLVM IR bloat, so this pass takes
// the associations from !sycl_aspects and then updates all the
// !sycl_used_aspects metadata to include the aspect names, which allows us
// to preserve these associations.
//===----------------------------------------------------------------------===//
//

#include "llvm/SYCLLowerIR/RecordSYCLAspectNames.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses RecordSYCLAspectNamesPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  SmallDenseMap<int64_t, Metadata *, 128> ValueToNameValuePairMD;
  if (NamedMDNode *Node = M.getNamedMetadata("sycl_aspects")) {
    for (MDNode *N : Node->operands()) {
      assert(N->getNumOperands() == 2 &&
             "Each operand of sycl_aspects must be a pair.");

      // The aspect's integral value is the second operand.
      auto *C = mdconst::extract<ConstantInt>(N->getOperand(1));
      ValueToNameValuePairMD[C->getSExtValue()] = N;
    }
  }

  auto &Ctx = M.getContext();
  const char *MetadataToProcess[] = {"sycl_used_aspects",
                                     "sycl_declared_aspects"};
  for (Function &F : M.functions()) {
    for (auto MetadataName : MetadataToProcess) {
      auto *MDNode = F.getMetadata(MetadataName);
      if (!MDNode)
        continue;

      // Change the metadata from {1, 2} to
      // a format like {{"cpu", 1}, {"gpu", 2}}
      SmallVector<Metadata *, 8> AspectNameValuePairs;
      for (const auto &MDOp : MDNode->operands()) {
        auto *C = mdconst::extract<ConstantInt>(MDOp);
        int64_t AspectValue = C->getSExtValue();
        if (auto It = ValueToNameValuePairMD.find(AspectValue);
            It != ValueToNameValuePairMD.end())
          AspectNameValuePairs.push_back(It->second);
        else
          AspectNameValuePairs.push_back(MDOp);
      }

      F.setMetadata(MetadataName, MDNode::get(Ctx, AspectNameValuePairs));
    }
  }

  return PreservedAnalyses::all();
}