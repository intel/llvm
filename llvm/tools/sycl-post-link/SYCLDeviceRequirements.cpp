//===----- SYCLDeviceRequirements.cpp - collect data for used aspects ----=-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLDeviceRequirements.h"
#include "ModuleSplitter.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/PropertySetIO.h"

#include <set>
#include <vector>

using namespace llvm;

void llvm::getSYCLDeviceRequirements(
    const module_split::ModuleDesc &MD,
    std::map<StringRef, util::PropertyValue> &Requirements) {
  auto ExtractIntegerFromMDNodeOperand = [=](const MDNode *N,
                                             unsigned OpNo) -> int32_t {
    Constant *C =
        cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
    return static_cast<int32_t>(C->getUniqueInteger().getSExtValue());
  };

  // { LLVM-IR metadata name , [SYCL/Device requirements] property name }, see:
  // https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OptionalDeviceFeatures.md#create-the-sycldevice-requirements-property-set
  // Scan the module and if the metadata is present fill the corresponing
  // property with metadata's aspects
  constexpr std::pair<const char *, const char *> ReqdMDs[] = {
      {"sycl_used_aspects", "aspects"},
      {"sycl_fixed_targets", "fixed_target"},
      {"reqd_work_group_size", "reqd_work_group_size"}};

  for (const auto &[MDName, MappedName] : ReqdMDs) {
    std::set<uint32_t> Values;
    for (const Function &F : MD.getModule()) {
      if (const MDNode *MDN = F.getMetadata(MDName)) {
        for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
          // Don't put internal aspects (with negative integer value) into the
          // requirements, they are used only for device image splitting.
          auto Val = ExtractIntegerFromMDNodeOperand(MDN, I);
          if (Val >= 0)
            Values.insert(Val);
        }
      }
    }

    // We don't need the "fixed_target" property if it's empty
    if (std::string(MDName) == "sycl_fixed_targets" && Values.empty())
      continue;
    Requirements[MappedName] =
        std::vector<uint32_t>(Values.begin(), Values.end());
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
      auto MDValue = ExtractIntegerFromMDNodeOperand(MDN, 0);
      assert(MDValue >= 0);
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
