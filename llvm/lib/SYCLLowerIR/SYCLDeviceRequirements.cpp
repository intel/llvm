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

SYCLDeviceRequirements
llvm::computeDeviceRequirements(const module_split::ModuleDesc &MD) {
  SYCLDeviceRequirements Reqs;
  // Process all functions in the module
  for (const Function &F : MD.getModule()) {
    if (auto *MDN = F.getMetadata("sycl_used_aspects")) {
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
        auto Val = ExtractSignedIntegerFromMDNodeOperand(MDN, I);
        // Don't put internal aspects (with negative integer value) into the
        // requirements, they are used only for device image splitting.
        if (Val >= 0)
          Reqs.Aspects.insert(Val);
      }
    }

    if (auto *MDN = F.getMetadata("sycl_fixed_targets")) {
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
        auto Val = ExtractUnsignedIntegerFromMDNodeOperand(MDN, I);
        Reqs.FixedTarget.insert(Val);
      }
    }

    if (auto *MDN = F.getMetadata("reqd_work_group_size")) {
      llvm::SmallVector<uint64_t, 3> NewReqdWorkGroupSize;
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I)
        NewReqdWorkGroupSize.push_back(
            ExtractUnsignedIntegerFromMDNodeOperand(MDN, I));
      if (!Reqs.ReqdWorkGroupSize.has_value())
        Reqs.ReqdWorkGroupSize = NewReqdWorkGroupSize;
    }

    if (auto *MDN = F.getMetadata("sycl_joint_matrix")) {
      auto Val = ExtractStringFromMDNodeOperand(MDN, 0);
      if (!Val.empty())
        Reqs.JointMatrix = Val;
    }

    if (auto *MDN = F.getMetadata("sycl_joint_matrix_mad")) {
      auto Val = ExtractStringFromMDNodeOperand(MDN, 0);
      if (!Val.empty())
        Reqs.JointMatrixMad = Val;
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
      assert(
          MDN->getNumOperands() == 1 &&
          "intel_reqd_sub_group_size metadata expects exactly one argument!");
      auto MDValue = ExtractUnsignedIntegerFromMDNodeOperand(MDN, 0);
      if (!Reqs.SubGroupSize)
        Reqs.SubGroupSize = MDValue;
      else
        assert(*Reqs.SubGroupSize == static_cast<uint32_t>(MDValue));
    }
  }
  return Reqs;
}

std::map<StringRef, util::PropertyValue> SYCLDeviceRequirements::asMap() const {
  std::map<StringRef, util::PropertyValue> Requirements;

  // For all properties except for "aspects", we'll only add the
  // value to the map if the corresponding value from
  // SYCLDeviceRequirements has a value/is non-empty.
  Requirements["aspects"] =
      std::vector<uint32_t>(Aspects.begin(), Aspects.end());

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

  if (SubGroupSize.has_value())
    Requirements["reqd_sub_group_size"] = *SubGroupSize;

  return Requirements;
}
