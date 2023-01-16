//==---------------------- TargetFusionInfo.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetFusionInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace llvm;

//
// TargetFusionInfo
//

TargetFusionInfo TargetFusionInfo::getTargetFusionInfo(llvm::Module *Mod) {
  llvm::Triple Tri(Mod->getTargetTriple());
  if (Tri.isNVPTX()) {
    return TargetFusionInfo(
        std::shared_ptr<NVPTXTargetFusionInfo>(new NVPTXTargetFusionInfo(Mod)));
  }
  if (Tri.isSPIRV()) {
    return TargetFusionInfo(
        std::shared_ptr<SPIRVTargetFusionInfo>(new SPIRVTargetFusionInfo(Mod)));
  }
  assert(false && "Unsupported target for fusion");
}

//
// SPIRVTargetFusionInfo
//

void SPIRVTargetFusionInfo::addKernelFunction(Function *KernelFunc) {
  KernelFunc->setCallingConv(CallingConv::SPIR_KERNEL);
}

ArrayRef<StringRef> SPIRVTargetFusionInfo::getKernelMetadataKeys() {
  static SmallVector<StringRef> Keys{
      {"kernel_arg_addr_space", "kernel_arg_access_qual", "kernel_arg_type",
       "kernel_arg_base_type", "kernel_arg_type_qual"}};
  return Keys;
}

void SPIRVTargetFusionInfo::postProcessKernel(Function *KernelFunc) {
  static constexpr auto ITTStartWrapper = "__itt_offload_wi_start_wrapper";
  static constexpr auto ITTFinishWrapper = "__itt_offload_wi_finish_wrapper";
  // Remove all existing calls of the ITT instrumentation functions. Insert new
  // ones in the entry block of the fused kernel and every exit block if the
  // functions are present in the module.
  // We cannot use the existing SPIRITTAnnotations pass, because that pass might
  // insert calls to functions not present in the module (e.g., ITT
  // instrumentations for barriers). As the JITed module is not linked with
  // libdevice anymore, the functions would remain unresolved and cause the
  // driver to fail.
  Function *StartWrapperFunc = LLVMMod->getFunction(ITTStartWrapper);
  Function *FinishWrapperFunc = LLVMMod->getFunction(ITTFinishWrapper);
  bool InsertWrappers =
      ((StartWrapperFunc && !StartWrapperFunc->isDeclaration()) &&
       (FinishWrapperFunc && !FinishWrapperFunc->isDeclaration()));
  auto *WrapperFuncTy = FunctionType::get(
      Type::getVoidTy(LLVMMod->getContext()), /*isVarArg*/ false);
  for (auto &BB : *KernelFunc) {
    for (auto Inst = BB.begin(); Inst != BB.end();) {
      if (auto *CB = dyn_cast<CallBase>(Inst)) {
        if (CB->getCalledFunction()->getName().starts_with("__itt_offload")) {
          Inst = Inst->eraseFromParent();
          continue;
        }
      }
      ++Inst;
    }
    if (InsertWrappers) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
        auto *WrapperCall =
            CallInst::Create(WrapperFuncTy, FinishWrapperFunc, "", RI);
        WrapperCall->setCallingConv(CallingConv::SPIR_FUNC);
      }
    }
  }
  if (InsertWrappers) {
    KernelFunc->getEntryBlock().getFirstInsertionPt();
    auto *WrapperCall =
        CallInst::Create(WrapperFuncTy, StartWrapperFunc, "",
                         &*KernelFunc->getEntryBlock().getFirstInsertionPt());
    WrapperCall->setCallingConv(CallingConv::SPIR_FUNC);
  }
}

void SPIRVTargetFusionInfo::createBarrierCall(IRBuilderBase &Builder,
                                              int BarrierFlags) {
  if (BarrierFlags == -1) {
    return;
  }
  assert((BarrierFlags == 1 || BarrierFlags == 2 || BarrierFlags == 3) &&
         "Invalid barrier flags");

  static const auto FnAttrs = AttributeSet::get(
      LLVMMod->getContext(),
      {Attribute::get(LLVMMod->getContext(), Attribute::AttrKind::Convergent),
       Attribute::get(LLVMMod->getContext(), Attribute::AttrKind::NoUnwind)});

  static constexpr StringLiteral N{"_Z22__spirv_ControlBarrierjjj"};

  Function *F = LLVMMod->getFunction(N);
  if (!F) {
    constexpr auto Linkage = GlobalValue::LinkageTypes::ExternalLinkage;

    auto *Ty = FunctionType::get(
        Builder.getVoidTy(),
        {Builder.getInt32Ty(), Builder.getInt32Ty(), Builder.getInt32Ty()},
        false /* isVarArg*/);

    F = Function::Create(Ty, Linkage, N, *LLVMMod);

    F->setAttributes(
        AttributeList::get(LLVMMod->getContext(), FnAttrs, {}, {}));
    F->setCallingConv(CallingConv::SPIR_FUNC);
  }

  // See
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-
  SmallVector<Value *> Args{
      Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
      Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
      Builder.getInt32(0x10 | (BarrierFlags % 2 == 1 ? 0x100 : 0x0) |
                       ((BarrierFlags >> 1 == 1 ? 0x200 : 0x0)))};

  auto *BarrierCallInst = Builder.CreateCall(F, Args);
  BarrierCallInst->setAttributes(
      AttributeList::get(LLVMMod->getContext(), FnAttrs, {}, {}));
  BarrierCallInst->setCallingConv(CallingConv::SPIR_FUNC);
}

//
// NVPTXTargetFusionInfo
//

void NVPTXTargetFusionInfo::notifyFunctionsDelete(
    llvm::ArrayRef<Function *> Funcs) {
  SmallPtrSet<Constant *, 8> DeletedFuncs{Funcs.begin(), Funcs.end()};
  SmallVector<MDNode *> ValidKernels;
  auto *OldAnnotations = LLVMMod->getNamedMetadata("nvvm.annotations");
  for (auto *Op : OldAnnotations->operands()) {
    if (auto *TOp = dyn_cast<MDTuple>(Op)) {
      if (auto *COp = dyn_cast_if_present<ConstantAsMetadata>(
              TOp->getOperand(0).get())) {
        if (!DeletedFuncs.contains(COp->getValue())) {
          ValidKernels.push_back(Op);
          // Add to the set to also remove duplicate entries.
          DeletedFuncs.insert(COp->getValue());
        }
      }
    }
  }
  LLVMMod->eraseNamedMetadata(OldAnnotations);
  auto *NewAnnotations = LLVMMod->getOrInsertNamedMetadata("nvvm.annotations");
  for (auto *Kernel : ValidKernels) {
    NewAnnotations->addOperand(Kernel);
  }
}

void NVPTXTargetFusionInfo::addKernelFunction(Function *KernelFunc) {
  auto *NVVMAnnotations = LLVMMod->getOrInsertNamedMetadata("nvvm.annotations");
  auto *MDOne = ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(LLVMMod->getContext()), 1));
  auto *MDKernelString = MDString::get(LLVMMod->getContext(), "kernel");
  auto *MDFunc = ConstantAsMetadata::get(KernelFunc);
  SmallVector<Metadata *, 3> KernelMD({MDFunc, MDKernelString, MDOne});
  auto *Tuple = MDTuple::get(LLVMMod->getContext(), KernelMD);
  NVVMAnnotations->addOperand(Tuple);
}

ArrayRef<StringRef> NVPTXTargetFusionInfo::getKernelMetadataKeys() {
  // FIXME: Check whether we need to take care of sycl_fixed_targets.
  static SmallVector<StringRef> Keys{{"kernel_arg_buffer_location",
                                      "kernel_arg_runtime_aligned",
                                      "kernel_arg_exclusive_ptr"}};
  return Keys;
}

void NVPTXTargetFusionInfo::createBarrierCall(IRBuilderBase &Builder,
                                              int BarrierFlags) {
  if (BarrierFlags == -1) {
    return;
  }
  // Emit a call to llvm.nvvm.barrier0. From the user manual of the NVPTX
  // backend: "The ‘@llvm.nvvm.barrier0()’ intrinsic emits a PTX bar.sync 0
  // instruction, equivalent to the __syncthreads() call in CUDA."
  Builder.CreateIntrinsic(Intrinsic::NVVMIntrinsics::nvvm_barrier0, {}, {});
}

//
// MetadataCollection
//

MetadataCollection::MetadataCollection(ArrayRef<StringRef> MDKeys)
    : Keys{MDKeys}, Collection(MDKeys.size()) {}

void MetadataCollection::collectFromFunction(
    llvm::Function *Func, const ArrayRef<bool> IsArgPresentMask) {
  for (auto &Key : Keys) {
    // TODO: Do we want to assert for the presence of the metadata here?
    if (auto *MD = Func->getMetadata(Key)) {
      for (auto MaskedOps : llvm::zip(IsArgPresentMask, MD->operands())) {
        if (std::get<0>(MaskedOps)) {
          Collection[Key].emplace_back(std::get<1>(MaskedOps).get());
        }
      }
    }
  }
}

void MetadataCollection::attachToFunction(llvm::Function *Func) {
  for (auto &Key : Keys) {
    // Attach a list of fused metadata for a kind to the fused function.
    auto *MDEntries = MDNode::get(Func->getContext(), Collection[Key]);
    Func->setMetadata(Key, MDEntries);
  }
}
