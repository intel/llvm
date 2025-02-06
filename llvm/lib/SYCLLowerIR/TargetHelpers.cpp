//===----------- TargetHelpers.cpp - Helpers for SYCL kernels ------------ ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions for processing SYCL kernels.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/TargetHelpers.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace llvm {
namespace TargetHelpers {

KernelCache::KernelPayload::KernelPayload(NamedMDNode *ModuleAnnotationsMD)
    : ModuleAnnotationsMD(ModuleAnnotationsMD) {}

bool KernelCache::isKernel(Function &F) const {
  return KernelData.contains(&F);
}

void KernelCache::handleReplacedWith(Function &OldF, Function &NewF) {
  assert(KernelData.contains(&OldF) && "Unknown kernel");
  if (auto &KP = KernelData[&OldF]; KP.hasAnnotations()) {
    // Make sure that all dependent annotation nodes are kept up to date.
    for (MDNode *D : KP.DependentMDs)
      D->replaceOperandWith(0, ConstantAsMetadata::get(&NewF));
  }
}

void KernelCache::handleNewCloneOf(Function &OldF, Function &NewF,
                                   bool KernelOnly) {
  assert(KernelData.contains(&OldF) && "Unknown kernel");
  if (auto &KP = KernelData[&OldF]; KP.hasAnnotations()) {
    if (!KernelOnly) {
      // Otherwise we'll need to clone all metadata, possibly dropping ones
      // which we can't assume are safe to clone.
      llvm_unreachable("Unimplemented cloning logic");
    }
  }
}

void KernelCache::populateKernels(Module &M) {
  Triple T(M.getTargetTriple());

  // AMDGPU kernels are identified by their calling convention, and don't have
  // any annotations.
  if (T.isAMDGCN()) {
    for (auto &F : M) {
      if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
        Kernels.push_back(&F);
        KernelData[&F] = KernelPayload{};
      }
    }
    return;
  }

  // NVPTX kernels are identified by their calling convention, and may have
  // annotations.
  if (T.isNVPTX()) {
    auto *AnnotationMetadata = M.getNamedMetadata("nvvm.annotations");
    for (auto &F : M) {
      if (F.getCallingConv() == CallingConv::PTX_Kernel) {
        Kernels.push_back(&F);
	KernelData[&F] = KernelPayload{AnnotationMetadata};
      }
    }
    // Early-exiting as there are no DependentMDNodes when no AnnotationMetadata.
    if (!AnnotationMetadata)
      return;

    // It is possible that the annotations node contains multiple pointers to
    // the same metadata, recognise visited ones.
    SmallSet<MDNode *, 4> Visited;
    DenseMap<Function *, SmallVector<MDNode *, 4>> DependentMDNodes;

    for (auto *MDN : AnnotationMetadata->operands()) {
      if (Visited.contains(MDN) || MDN->getNumOperands() % 2 != 1)
        continue;

      Visited.insert(MDN);

      // Get a pointer to the entry point function from the metadata.
      const MDOperand &FuncOperand = MDN->getOperand(0);
      if (!FuncOperand)
        continue;

      if (auto *FuncConstant = dyn_cast<ConstantAsMetadata>(FuncOperand))
        if (auto *Func = dyn_cast<Function>(FuncConstant->getValue()))
          if (Func->getCallingConv() == CallingConv::PTX_Kernel)
             DependentMDNodes[Func].push_back(MDN);
    }
      
    // We need to match non-kernel metadata nodes using the kernel name to the
    // kernel nodes. To avoid checking matched nodes multiple times keep track
    // of handled entries.
    SmallPtrSet<MDNode *, 4> HandledNodes;
    for (auto &[F, KP] : KernelData) {
      for (MDNode *DepMDN : DependentMDNodes[F]) {
        if (HandledNodes.insert(DepMDN).second)
          KP.DependentMDs.push_back(DepMDN);
      }
    }
  }
}

bool isSYCLDevice(const Module &M) {
  if (auto *Flag = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("sycl-device"))) {
    return Flag->getZExtValue() == 1;
  }
  return false;
}

} // namespace TargetHelpers
} // namespace llvm
