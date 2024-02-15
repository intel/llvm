//===----- SYCLDeviceRequirements.cpp - collect data for used aspects ----=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLDeviceRequirements.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLLowerIR/ModuleSplitter.h"
#include "llvm/Support/PropertySetIO.h"

#include <set>
#include <vector>

using namespace llvm;

static int64_t ExtractSignedIntegerFromMDNodeOperand(const MDNode *N,
                                                     unsigned OpNo) {
  Constant *C = cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
  return C->getUniqueInteger().getSExtValue();
}
static uint64_t ExtractUnsignedIntegerFromMDNodeOperand(const MDNode *N,
                                                        unsigned OpNo) {
  Constant *C = cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
  return C->getUniqueInteger().getZExtValue();
}
static llvm::StringRef ExtractStringFromMDNodeOperand(const MDNode *N,
                                                      unsigned OpNo) {
  MDString *S = cast<llvm::MDString>(N->getOperand(OpNo).get());
  return S->getString();
}

SYCLDeviceRequirements::SYCLDeviceRequirements(
    const module_split::ModuleDesc &MD) {
  // Process all functions in the module
  for (const Function &F : MD.getModule()) {
    if (auto *MDN = F.getMetadata("sycl_used_aspects")) {
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
        auto Val = ExtractSignedIntegerFromMDNodeOperand(MDN, I);
        // Don't put internal aspects (with negative integer value) into the
        // requirements, they are used only for device image splitting.
        if (Val >= 0)
          Aspects.insert(Val);
      }
    }

    if (auto *MDN = F.getMetadata("sycl_fixed_targets")) {
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
        auto Val = ExtractUnsignedIntegerFromMDNodeOperand(MDN, I);
        FixedTarget.insert(Val);
      }
    }

    if (auto *MDN = F.getMetadata("reqd_work_group_size")) {
      llvm::SmallVector<uint64_t, 3> NewReqdWorkGroupSize;
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I)
        NewReqdWorkGroupSize.push_back(
            ExtractUnsignedIntegerFromMDNodeOperand(MDN, I));
      if (!ReqdWorkGroupSize.has_value())
        ReqdWorkGroupSize = NewReqdWorkGroupSize;
    }

    if (auto *MDN = F.getMetadata("sycl_joint_matrix")) {
      auto Val = ExtractStringFromMDNodeOperand(MDN, 0);
      if (!Val.empty())
        JointMatrix = Val;
    }

    if (auto *MDN = F.getMetadata("sycl_joint_matrix_mad")) {
      auto Val = ExtractStringFromMDNodeOperand(MDN, 0);
      if (!Val.empty())
        JointMatrixMad = Val;
    }
  }

  // Process just the entry points in the module
  for (const Function *F : MD.entries()) {
    if (auto *MDN = F->getMetadata("intel_reqd_sub_group_size")) {
      // There should only be at most one function with
      // intel_reqd_sub_group_size metadata when considering the entry
      // points of a module, but not necessarily when considering all the
      // functions of a module: an entry point with a
      // intel_reqd_sub_group_size can call an ESIMD function through
      // invoke_esimd, and that function has intel_reqd_sub_group_size=1,
      // which is valid.
      assert(MDN->getNumOperands() == 1);
      auto MDValue = ExtractUnsignedIntegerFromMDNodeOperand(MDN, 0);
      if (!SubGroupSize)
        SubGroupSize = MDValue;
      else
        assert(*SubGroupSize == static_cast<uint32_t>(MDValue));
    }
  }
}

std::map<StringRef, util::PropertyValue> SYCLDeviceRequirements::asMap() const {
  std::map<StringRef, util::PropertyValue> Requirements;

  Requirements["aspects"] =
      std::vector<uint32_t>(Aspects.begin(), Aspects.end());

  // We don't need the "fixed_target" property if it's empty
  if (!FixedTarget.empty())
    Requirements["fixed_target"] =
        std::vector<uint32_t>(FixedTarget.begin(), FixedTarget.end());

  // TODO: Before intel/llvm#10620, the reqd_work_group_size attribute
  // stores its values as uint32_t, but this needed to be expanded to
  // uint64_t.  However, this change did not happen in ABI-breaking
  // window, so we attach the required work-group size as the
  // reqd_work_group_size_uint64_t attribute. At the next ABI-breaking
  // window, this can be changed back to reqd_work_group_size.
  if (ReqdWorkGroupSize.has_value())
    Requirements["reqd_work_group_size_uint64_t"] = *ReqdWorkGroupSize;

  if (JointMatrix.has_value())
    Requirements["joint_matrix"] = *JointMatrix;

  if (JointMatrixMad.has_value())
    Requirements["joint_matrix_mad"] = *JointMatrixMad;

  // Do not attach reqd_sub_group_size if there is no attached metadata
  if (SubGroupSize.has_value())
    Requirements["reqd_sub_group_size"] = *SubGroupSize;

  return Requirements;
}
