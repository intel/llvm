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
  std::set<uint32_t> Aspects;
  auto FindAspectsByMDName = [&](const Function &F, std::string MDName) {
    if (const MDNode *MDN = F.getMetadata(MDName))
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
        Constant *C =
            cast<ConstantAsMetadata>(MDN->getOperand(I).get())->getValue();
        Aspects.insert(
            static_cast<uint32_t>(C->getUniqueInteger().getZExtValue()));
      }
  };

  // Scan the module and if the metadata is present fill the corresponing
  // property with metadata's aspects
  for (const Function &F : M) {
    FindAspectsByMDName(F, "sycl_used_aspects");
    FindAspectsByMDName(F, "sycl_declared_aspects");
  }
  Requirements["aspects"] =
      std::vector<uint32_t>(Aspects.begin(), Aspects.end());

  Aspects.clear();
  for (const Function &F : M)
    FindAspectsByMDName(F, "sycl_fixed_targets");
  // We don't need the "fixed_target" property if it's empty
  if (!Aspects.empty())
    Requirements["fixed_target"] =
        std::vector<uint32_t>(Aspects.begin(), Aspects.end());
}
