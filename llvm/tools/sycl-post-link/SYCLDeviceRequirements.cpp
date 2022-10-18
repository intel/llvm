//===----- SYCLDeviceRequirements.cpp - collect data for used aspects ----=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLDeviceRequirements.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <set>
#include <vector>

using namespace llvm;

void llvm::getSYCLDeviceRequirements(
    const Module &M, std::map<StringRef, std::vector<uint32_t>> &Requirements) {
  auto ExtractIntegerFromMDNodeOperand = [=](const MDNode *N,
                                             unsigned OpNo) -> unsigned {
    Constant *C =
        cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
    return static_cast<uint32_t>(C->getUniqueInteger().getZExtValue());
  };

  std::vector<std::pair<std::string, const char *>> ReqdMDs = {
      {"sycl_used_aspects", "aspects"}, {"sycl_fixed_targets", "fixed_target"}};

  for (const auto &MD : ReqdMDs) {
    std::set<uint32_t> Aspects;
    for (const Function &F : M) {
      if (F.hasMetadata(MD.first)) {
        const MDNode *MDN = F.getMetadata(MD.first);
        for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I)
          Aspects.insert(ExtractIntegerFromMDNodeOperand(MDN, I));
      }
    }
    // We don't need the "fixed_target" property if it's empty
    if (MD.first == "sycl_fixed_targets" && Aspects.empty())
      continue;
    Requirements[MD.second] =
        std::vector<uint32_t>(Aspects.begin(), Aspects.end());
  }
}
