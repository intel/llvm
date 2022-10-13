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

std::map<StringRef, std::vector<uint32_t>>
llvm::getSYCLDeviceRequirements(const Module &M) {
  std::map<StringRef, std::vector<uint32_t>> Result;
  auto ExtractIntegerFromMDNodeOperand = [=](const MDNode *N,
                                             unsigned OpNo) -> unsigned {
    Constant *C =
        cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
    return static_cast<uint32_t>(C->getUniqueInteger().getZExtValue());
  };
  std::set<uint32_t> Aspects;
  for (const Function &F : M) {
    if (!F.hasMetadata("sycl_used_aspects"))
      continue;

    const MDNode *MD = F.getMetadata("sycl_used_aspects");
    for (size_t I = 0, E = MD->getNumOperands(); I < E; ++I) {
      Aspects.insert(ExtractIntegerFromMDNodeOperand(MD, I));
    }
  }

  Result["aspects"] = std::vector<uint32_t>(Aspects.begin(), Aspects.end());
  return Result;
}
