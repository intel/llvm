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
    if (KernelOnly) {
      // We know this is a kernel, so add a single "kernel" annotation.
      auto &Ctx = OldF.getContext();
      Metadata *NewKernelMD[] = {
          ConstantAsMetadata::get(&NewF), MDString::get(Ctx, "kernel"),
          ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1))};
      KP.ModuleAnnotationsMD->addOperand(MDNode::get(Ctx, NewKernelMD));
    } else {
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

  // NVPTX kernels are identified by the global annotations metadata.
  if (T.isNVPTX()) {
    // Access `nvvm.annotations` to determine which functions are kernel
    // entry points.
    auto *AnnotationMetadata = M.getNamedMetadata("nvvm.annotations");
    // No kernels in the module, early exit.
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

      // Kernel entry points are identified using metadata nodes of the form:
      //   !X = !{<function>[, !"kind", i32 X]+}
      // Where "kind" == "kernel" and X == 1.
      bool IsKernel = false;
      for (size_t I = 1, E = MDN->getNumOperands() - 1; I < E && !IsKernel;
           I += 2) {
        if (auto *Type = dyn_cast<MDString>(MDN->getOperand(I)))
          if (Type->getString() == "kernel") {
            if (auto *Val =
                    mdconst::dyn_extract<ConstantInt>(MDN->getOperand(I + 1)))
              IsKernel = Val->getZExtValue() == 1;
          }
      }

      // Get a pointer to the entry point function from the metadata.
      const MDOperand &FuncOperand = MDN->getOperand(0);
      if (!FuncOperand)
        continue;

      if (auto *FuncConstant = dyn_cast<ConstantAsMetadata>(FuncOperand)) {
        if (auto *Func = dyn_cast<Function>(FuncConstant->getValue())) {
          if (IsKernel && !KernelData.contains(Func)) {
            Kernels.push_back(Func);
            KernelData[Func] = KernelPayload{AnnotationMetadata};
          }
          DependentMDNodes[Func].push_back(MDN);
        }
      }
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
