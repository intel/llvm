//===-- SanitizerKernelMetadata.cpp - fix kernel medatadata for sanitizer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass fixes attributes and metadata of global variable
// "__AsanKernelMetadata" or "__MsanKernelMetadata".
// We treat "KernelMetadata" as a device global variable, so that it
// can be read by runtime.
// "spirv.Decorations" is removed by llvm-link, so we add it here again.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SanitizerPostOptimizer.h"
#include "llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"

#define DEBUG_TYPE "SanitizerPostOptimizer"

using namespace llvm;

namespace llvm {

constexpr StringRef SPIRV_DECOR_MD_KIND = "spirv.Decorations";
constexpr uint32_t SPIRV_HOST_ACCESS_DECOR = 6147;

struct EliminateDeadCheck : public InstVisitor<EliminateDeadCheck> {
  void visitCallInst(CallInst &CI) {
    // If the shadow value is constant zero, the check instruction can be safely
    // erased.
    auto *Func = CI.getCalledFunction();
    if (!Func)
      return;
    auto FuncName = Func->getName();
    if (!FuncName.contains("__msan_maybe_warning_"))
      return;
    auto *Shadow = CI.getArgOperand(0);
    if (isa<ConstantInt>(Shadow) && cast<ConstantInt>(Shadow)->isNullValue())
      InstToErase.push_back(&CI);
  }

  bool eraseDeadCheck() {
    bool Changed = !InstToErase.empty();
    for (auto *CI : InstToErase)
      CI->eraseFromParent();
    InstToErase.clear();
    return Changed;
  }

private:
  SmallVector<CallInst *, 8> InstToErase;
};

static bool FixSanitizerKernelMetadata(Module &M) {
  SmallVector<GlobalVariable *, 4> KernelMetadatas;
  for (GlobalVariable &GV : M.globals()) {
    auto GVName = GV.getName();
    if (GVName.starts_with("__AsanKernelMetadata") ||
        GVName.starts_with("__MsanKernelMetadata") ||
        GVName.starts_with("__TsanKernelMetadata")) {
      KernelMetadatas.push_back(&GV);
    }
  }

  if (KernelMetadatas.empty())
    return false;

  auto &DL = M.getDataLayout();
  auto &Ctx = M.getContext();

  // Fix device global type, by wrapping a structure type
  for (GlobalVariable *KernelMetadata : KernelMetadatas) {
    assert(KernelMetadata->getValueType()->isArrayTy());

    auto *KernelMetadataOld = KernelMetadata;

    StructType *StructTypeWithArray = StructType::create(Ctx);
    StructTypeWithArray->setBody(KernelMetadataOld->getValueType());

    KernelMetadata = new GlobalVariable(
        M, StructTypeWithArray, false, GlobalValue::ExternalLinkage,
        ConstantStruct::get(StructTypeWithArray,
                            KernelMetadataOld->getInitializer()),
        "", nullptr, GlobalValue::NotThreadLocal, 1); // Global AddressSpace
    KernelMetadata->takeName(KernelMetadataOld);
    KernelMetadata->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    KernelMetadata->setDSOLocal(true);
    KernelMetadata->copyAttributesFrom(KernelMetadataOld);
    KernelMetadataOld->eraseFromParent();

    // Fix attributes
    KernelMetadata->addAttribute(
        "sycl-device-global-size",
        std::to_string(DL.getTypeAllocSize(KernelMetadata->getValueType())));

    // Fix metadata
    unsigned MDKindID = Ctx.getMDKindID(SPIRV_DECOR_MD_KIND);

    SmallVector<Metadata *, 1> MDOps;

    SmallVector<Metadata *, 3> MD;
    auto *Ty = Type::getInt32Ty(Ctx);
    MD.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Ty, APInt(32, SPIRV_HOST_ACCESS_DECOR))));
    MD.push_back(
        ConstantAsMetadata::get(Constant::getIntegerValue(Ty, APInt(32, 0))));
    MD.push_back(MDString::get(Ctx, "_Z20__SanitizerKernelMetadata"));

    MDOps.push_back(MDNode::get(Ctx, MD));

    KernelMetadata->addMetadata(MDKindID, *MDNode::get(Ctx, MDOps));
  }

  return true;
}

// SPIR-V does not support weak_odr linkage. Change sanitizer configuration
// globals to linkonce_odr.
static void FixWeakGlobalVariable(Module &M) {
  if (GlobalVariable *GV = M.getNamedGlobal("__asan_check_shadow_bounds"))
    GV->setLinkage(GlobalVariable::LinkOnceODRLinkage);
  if (GlobalVariable *GV = M.getNamedGlobal("__msan_track_origins"))
    GV->setLinkage(GlobalVariable::LinkOnceODRLinkage);
}

PreservedAnalyses SanitizerPostOptimizerPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  if (!FixSanitizerKernelMetadata(M))
    return PreservedAnalyses::all();

  if (sycl::isModuleUsingMsan(M)) {
    EliminateDeadCheck V;
    V.visit(M);
    V.eraseDeadCheck();
  }

  FixWeakGlobalVariable(M);

  return PreservedAnalyses::none();
}

} // namespace llvm
