// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/attributes.h>
#include <compiler/utils/device_info.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/vectorization_factor.h>
#include <compiler/utils/verify_reqd_sub_group_size_pass.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace {
class DiagnosticInfoReqdSGSize : public DiagnosticInfoWithLocationBase {
  uint32_t SGSize;

 public:
  static int DK_FailedReqdSGSize;
  static int DK_UnsupportedReqdSGSize;

  DiagnosticInfoReqdSGSize(const Function &F, uint32_t SGSize, int Kind)
      : DiagnosticInfoWithLocationBase(static_cast<DiagnosticKind>(Kind),
                                       DS_Error, F, F.getSubprogram()),
        SGSize(SGSize) {
    assert(Kind == DK_FailedReqdSGSize || Kind == DK_UnsupportedReqdSGSize);
  }

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_FailedReqdSGSize ||
           DI->getKind() == DK_UnsupportedReqdSGSize;
  }

  void print(DiagnosticPrinter &DP) const override {
    DP << getLocationStr() << ": kernel '" << this->getFunction().getName()
       << "' has required sub-group size " << SGSize;
    if (getKind() == DK_FailedReqdSGSize) {
      DP << " but the compiler was unable to sastify this constraint";
    } else {
      DP << " which is not supported by this device";
    }
  }
};

int DiagnosticInfoReqdSGSize::DK_FailedReqdSGSize =
    getNextAvailablePluginDiagnosticKind();
int DiagnosticInfoReqdSGSize::DK_UnsupportedReqdSGSize =
    getNextAvailablePluginDiagnosticKind();
}  // namespace

namespace compiler {
namespace utils {
PreservedAnalyses VerifyReqdSubGroupSizeLegalPass::run(
    Module &M, ModuleAnalysisManager &AM) {
  auto &DI = AM.getResult<DeviceInfoAnalysis>(M);
  const auto &SGSizes = DI.reqd_sub_group_sizes;
  for (auto &F : M) {
    const auto ReqdSGSize = getReqdSubgroupSize(F);
    if (!ReqdSGSize) {
      continue;
    }
    // If this sub-group size is not supported by the device, we can emit a
    // diagnostic at compile-time.
    if (std::find(SGSizes.begin(), SGSizes.end(), *ReqdSGSize) ==
        SGSizes.end()) {
      M.getContext().diagnose(DiagnosticInfoReqdSGSize(
          F, *ReqdSGSize, DiagnosticInfoReqdSGSize::DK_UnsupportedReqdSGSize));
    }
  }
  return PreservedAnalyses::all();
}

PreservedAnalyses VerifyReqdSubGroupSizeSatisfiedPass::run(
    Module &M, ModuleAnalysisManager &) {
  for (auto &F : M) {
    // We only check kernel entry points
    if (!isKernelEntryPt(F)) {
      continue;
    }

    const auto ReqdSGSize = getReqdSubgroupSize(F);
    if (!ReqdSGSize) {
      continue;
    }

    auto CurrSGSize = VectorizationFactor::getFixedWidth(
        compiler::utils::getMuxSubgroupSize(F));
    if (auto VeczInfo = parseVeczToOrigFnLinkMetadata(F)) {
      CurrSGSize = VeczInfo->second.vf * CurrSGSize.getKnownMin();
    }

    if (CurrSGSize != ReqdSGSize) {
      M.getContext().diagnose(DiagnosticInfoReqdSGSize(
          F, *ReqdSGSize, DiagnosticInfoReqdSGSize::DK_FailedReqdSGSize));
    }
  }

  return PreservedAnalyses::all();
}

}  // namespace utils
}  // namespace compiler
