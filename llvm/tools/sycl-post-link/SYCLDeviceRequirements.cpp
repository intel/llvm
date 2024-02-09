//===----- SYCLDeviceRequirements.cpp - collect data for used aspects ----=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLDeviceRequirements.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLLowerIR/ModuleSplitter.h"
#include "llvm/Support/PropertySetIO.h"

#include <set>
#include <vector>

using namespace llvm;

void llvm::getSYCLDeviceRequirements(
    const module_split::ModuleDesc &MD,
    std::map<StringRef, util::PropertyValue> &Requirements) {
  auto ExtractSignedIntegerFromMDNodeOperand = [=](const MDNode *N,
                                                   unsigned OpNo) -> int64_t {
    Constant *C =
        cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
    return C->getUniqueInteger().getSExtValue();
  };

  auto ExtractUnsignedIntegerFromMDNodeOperand =
      [=](const MDNode *N, unsigned OpNo) -> uint64_t {
    Constant *C =
        cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
    return C->getUniqueInteger().getZExtValue();
  };

  // { LLVM-IR metadata name , [SYCL/Device requirements] property name }, see:
  // https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OptionalDeviceFeatures.md#create-the-sycldevice-requirements-property-set
  // Scan the module and if the metadata is present fill the corresponing
  // property with metadata's aspects
  constexpr std::pair<const char *, const char *> ReqdMDs[] = {
      {"sycl_used_aspects", "aspects"}, {"sycl_fixed_targets", "fixed_target"}};

  for (const auto &[MDName, MappedName] : ReqdMDs) {
    std::set<uint32_t> Values;
    for (const Function &F : MD.getModule()) {
      if (const MDNode *MDN = F.getMetadata(MDName)) {
        for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
          if (std::string(MDName) == "sycl_used_aspects") {
            // Don't put internal aspects (with negative integer value) into the
            // requirements, they are used only for device image splitting.
            auto Val = ExtractSignedIntegerFromMDNodeOperand(MDN, I);
            if (Val >= 0)
              Values.insert(Val);
          } else {
            Values.insert(ExtractUnsignedIntegerFromMDNodeOperand(MDN, I));
          }
        }
      }
    }

    // We don't need the "fixed_target" property if it's empty
    if (std::string(MDName) == "sycl_fixed_targets" && Values.empty())
      continue;
    Requirements[MappedName] =
        std::vector<uint32_t>(Values.begin(), Values.end());
  }

  std::optional<llvm::SmallVector<uint64_t, 3>> ReqdWorkGroupSize;
  for (const Function &F : MD.getModule()) {
    if (const MDNode *MDN = F.getMetadata("reqd_work_group_size")) {
      llvm::SmallVector<uint64_t, 3> NewReqdWorkGroupSize;
      for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I)
        NewReqdWorkGroupSize.push_back(
            ExtractUnsignedIntegerFromMDNodeOperand(MDN, I));
      if (!ReqdWorkGroupSize)
        ReqdWorkGroupSize = NewReqdWorkGroupSize;
    }
  }

  // TODO: Before intel/llvm#10620, the reqd_work_group_size attribute
  // stores its values as uint32_t, but this needed to be expanded to
  // uint64_t.  However, this change did not happen in ABI-breaking
  // window, so we attach the required work-group size as the
  // reqd_work_group_size_uint64_t attribute. At the next ABI-breaking
  // window, this can be changed back to reqd_work_group_size.
  if (ReqdWorkGroupSize)
    Requirements["reqd_work_group_size_uint64_t"] = *ReqdWorkGroupSize;

  auto ExtractStringFromMDNodeOperand =
      [=](const MDNode *N, unsigned OpNo) -> llvm::SmallString<256> {
    MDString *S = cast<llvm::MDString>(N->getOperand(OpNo).get());
    return S->getString();
  };

  // { LLVM-IR metadata name , [SYCL/Device requirements] property name }, see:
  // https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OptionalDeviceFeatures.md#create-the-sycldevice-requirements-property-set
  // Scan the module and if the metadata is present fill the corresponing
  // property with metadata's aspects
  constexpr std::pair<const char *, const char *> MatrixMDs[] = {
      {"sycl_joint_matrix", "joint_matrix"},
      {"sycl_joint_matrix_mad", "joint_matrix_mad"}};

  for (const auto &[MDName, MappedName] : MatrixMDs) {
    llvm::SmallString<256> Val;
    for (const Function &F : MD.getModule())
      if (const MDNode *MDN = F.getMetadata(MDName))
        Val = ExtractStringFromMDNodeOperand(
            MDN, 0); // there is always only one operand
    if (Val.empty())
      continue;
    Requirements[MappedName] = Val;
  }

  // There should only be at most one function with
  // intel_reqd_sub_group_size metadata when considering the entry
  // points of a module, but not necessarily when considering all the
  // functions of a module: an entry point with a
  // intel_reqd_sub_group_size can call an ESIMD function through
  // invoke_esimd, and that function has intel_reqd_sub_group_size=1,
  // which is valid.
  std::optional<uint32_t> SubGroupSize;
  for (const Function *F : MD.entries()) {
    if (auto *MDN = F->getMetadata("intel_reqd_sub_group_size")) {
      assert(MDN->getNumOperands() == 1);
      auto MDValue = ExtractUnsignedIntegerFromMDNodeOperand(MDN, 0);
      if (!SubGroupSize)
        SubGroupSize = MDValue;
      else
        assert(*SubGroupSize == static_cast<uint32_t>(MDValue));
    }
  }
  // Do not attach reqd_sub_group_size if there is no attached metadata
  if (SubGroupSize)
    Requirements["reqd_sub_group_size"] = *SubGroupSize;
}
