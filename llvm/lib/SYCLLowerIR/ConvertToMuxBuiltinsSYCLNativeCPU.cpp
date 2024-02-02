//===-- ConvertToMuxBuiltinsSYCLNativeCPU.cpp - Convert to Mux Builtins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts SPIRV builtins to Mux builtins used by the oneAPI Construction
// Kit for SYCL Native CPU
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ConvertToMuxBuiltinsSYCLNativeCPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <map>

using namespace llvm;

namespace {

static void fixFunctionAttributes(Function *F) {
  // The frame-pointer=all and the "byval" attributes lead to code generation
  // that conflicts with the Kernel declaration that we emit in the Native CPU
  // helper header (in which all the kernel argument are void* or scalars).
  auto AttList = F->getAttributes();
  for (unsigned ArgNo = 0; ArgNo < F->getFunctionType()->getNumParams();
       ArgNo++) {
    if (AttList.hasParamAttr(ArgNo, Attribute::AttrKind::ByVal)) {
      AttList = AttList.removeParamAttribute(F->getContext(), ArgNo,
                                             Attribute::AttrKind::ByVal);
    }
  }
  F->setAttributes(AttList);
  F->addFnAttr("frame-pointer", "none");
}

static inline bool isForVisualStudio(StringRef TripleStr) {
  llvm::Triple Triple(TripleStr);
  return Triple.isKnownWindowsMSVCEnvironment();
}

static constexpr char SPIRVBarrier[] = "_Z22__spirv_ControlBarrierjjj";
static constexpr char SPIRVBarrierWin[] = "?__spirv_ControlBarrier@@YAXIII@Z";
static constexpr char MuxBarrier[] = "__mux_work_group_barrier";

Function *getMuxBarrierFunc(Module &M) {
  // void __mux_work_group_barrier(i32 %id, i32 %scope, i32 %semantics)
  LLVMContext &Ctx = M.getContext();
  auto *Int32Ty = Type::getInt32Ty(Ctx);
  static auto *MuxFTy = FunctionType::get(Type::getVoidTy(Ctx),
                                          {Int32Ty, Int32Ty, Int32Ty}, false);
  auto F = M.getOrInsertFunction(MuxBarrier, MuxFTy);
  return cast<Function>(F.getCallee());
}

static constexpr const char *MuxKernelAttrName = "mux-kernel";

void setIsKernelEntryPt(Function &F) {
  F.addFnAttr(MuxKernelAttrName, "entry-point");
}

struct InstReplacer {
  typedef std::pair<Instruction *, Instruction *> T;
  SmallVector<T> ToRemove;
  void push_back(T a) { ToRemove.push_back(a); }
  void apply() { // todo
    for (auto &El : ToRemove) {
      auto OldI = El.first;
      auto NewI = El.second;
      OldI->replaceAllUsesWith(NewI);
      OldI->eraseFromParent();
    }
  }
};

static bool replaceBarriers(Module &M, bool VSMangling) {
  // DPC++ emits
  //__spirv_ControlBarrier(__spv::Scope Execution, __spv::Scope Memory,
  //                       uint32_t Semantics) noexcept;
  // OCK expects  void __mux_work_group_barrier(i32 %id, i32 %scope, i32
  // %semantics)
  // __spv::Scope is
  // enum Flag : uint32_t {
  //   CrossDevice = 0,
  //   Device = 1,
  //   Workgroup = 2,
  //   Subgroup = 3,
  //   Invocation = 4,
  // };
  auto *SPIRVBarrierFunc =
      M.getFunction(VSMangling ? SPIRVBarrierWin : SPIRVBarrier);
  if (!SPIRVBarrierFunc) {
    // No barriers are found, just return
    return false;
  }
  static auto *MuxBarrierFunc = getMuxBarrierFunc(M);
  InstReplacer ToRemove;
  auto *Zero = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
  for (auto &Use : SPIRVBarrierFunc->uses()) {
    auto *I = dyn_cast<CallInst>(Use.getUser());
    if (!I)
      report_fatal_error("Unsupported Value in SYCL Native CPU\n");
    SmallVector<Value *, 3> Args{Zero, I->getArgOperand(0),
                                 I->getArgOperand(2)}; // todo: check how the
                                                       // args map to each other
    auto *NewI = CallInst::Create(MuxBarrierFunc->getFunctionType(),
                                  MuxBarrierFunc, Args, "", I);
    ToRemove.push_back(std::pair(I, NewI));
  }

  ToRemove.apply();

  SPIRVBarrierFunc->eraseFromParent();

  return true;
}

} // namespace

PreservedAnalyses
ConvertToMuxBuiltinsSYCLNativeCPUPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      fixFunctionAttributes(&F);
      setIsKernelEntryPt(F);
    }
  }
  const bool VisualStudioMangling = isForVisualStudio(M.getTargetTriple());

  ModuleChanged |= replaceBarriers(M, VisualStudioMangling);
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
