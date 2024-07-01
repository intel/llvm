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
#include <compiler/utils/encode_kernel_metadata_pass.h>
#include <compiler/utils/metadata.h>
#include <llvm/IR/Module.h>

using namespace llvm;

PreservedAnalyses compiler::utils::TransferKernelMetadataPass::run(
    Module &M, ModuleAnalysisManager &) {
  SmallVector<KernelInfo, 4> Kernels;
  populateKernelList(M, Kernels);

  for (const auto &Kernel : Kernels) {
    if (auto *F = M.getFunction(Kernel.Name)) {
      setOrigFnName(*F);
      setIsKernelEntryPt(*F);
      if (Kernel.ReqdWGSize) {
        encodeLocalSizeMetadata(*F, *Kernel.ReqdWGSize);
      }
    }
  }

  return PreservedAnalyses::all();
}

PreservedAnalyses compiler::utils::EncodeKernelMetadataPass::run(
    Module &M, ModuleAnalysisManager &) {
  if (auto *F = M.getFunction(KernelName)) {
    setOrigFnName(*F);
    setIsKernelEntryPt(*F);
    if (LocalSizes) {
      encodeLocalSizeMetadata(*F, *LocalSizes);
    }
  }
  return PreservedAnalyses::all();
}
