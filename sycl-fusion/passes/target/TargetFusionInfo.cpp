//==---------------------- TargetFusionInfo.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetFusionInfo.h"

#include "Kernel.h"
#include "NDRangesHelper.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/TargetParser/Triple.h"

template <typename ForwardIt, typename KeyTy>
static ForwardIt mapArrayLookup(ForwardIt Begin, ForwardIt End,
                                const KeyTy &Key) {
  return std::lower_bound(
      Begin, End, Key,
      [](const auto &Entry, const auto &Key) { return Entry.first < Key; });
}

template <typename ForwardIt, typename KeyTy>
static auto mapArrayLookupValue(ForwardIt Begin, ForwardIt End,
                                const KeyTy &Key)
    -> std::optional<decltype(Begin->second)> {
  const auto Iter = mapArrayLookup(Begin, End, Key);
  if (Iter != End && Iter->first == Key)
    return Iter->second;
  return {};
}

namespace llvm {

using jit_compiler::BuiltinKind;
using jit_compiler::NDRange;
using jit_compiler::Remapper;

using jit_compiler::requireIDRemapping;

class TargetFusionInfoImpl {

public:
  explicit TargetFusionInfoImpl(llvm::Module *Mod) : LLVMMod{Mod} {};

  virtual ~TargetFusionInfoImpl() = default;

  virtual void notifyFunctionsDelete(
      [[maybe_unused]] llvm::ArrayRef<Function *> Funcs) const {}

  virtual void addKernelFunction([[maybe_unused]] Function *KernelFunc) const {}

  virtual void postProcessKernel([[maybe_unused]] Function *KernelFunc) const {}

  virtual ArrayRef<StringRef> getKernelMetadataKeys() const { return {}; }

  virtual ArrayRef<StringRef> getUniformKernelAttributes() const { return {}; }

  virtual void createBarrierCall(IRBuilderBase &Builder,
                                 int BarrierFlags) const = 0;

  virtual unsigned getPrivateAddressSpace() const = 0;

  virtual unsigned getLocalAddressSpace() const = 0;

  virtual void
  updateAddressSpaceMetadata([[maybe_unused]] Function *KernelFunc,
                             [[maybe_unused]] ArrayRef<size_t> LocalSize,
                             [[maybe_unused]] unsigned AddressSpace) const {}

  virtual std::optional<BuiltinKind> getBuiltinKind(Function *F) const = 0;

  virtual bool shouldRemap(BuiltinKind K, const NDRange &SrcNDRange,
                           const NDRange &FusedNDRange) const = 0;

  virtual bool isSafeToNotRemapBuiltin(Function *F) const = 0;

  virtual unsigned getIndexSpaceBuiltinBitwidth() const = 0;

  virtual void setMetadataForGeneratedFunction(Function *F) const = 0;

  virtual Value *getGlobalIDWithoutOffset(IRBuilderBase &Builder,
                                          const NDRange &FusedNDRange,
                                          uint32_t Idx) const = 0;

  virtual Function *
  createRemapperFunction(const Remapper &R, BuiltinKind K, StringRef OrigName,
                         StringRef Name, Module *M, const NDRange &SrcNDRange,
                         const NDRange &FusedNDRange) const = 0;

protected:
  llvm::Module *LLVMMod;
};

namespace {

//
// SPIRVTargetFusionInfo
//
class SPIRVTargetFusionInfo : public TargetFusionInfoImpl {
public:
  using TargetFusionInfoImpl::TargetFusionInfoImpl;

  void addKernelFunction(Function *KernelFunc) const override {
    KernelFunc->setCallingConv(CallingConv::SPIR_KERNEL);
  }

  ArrayRef<StringRef> getKernelMetadataKeys() const override {
    // NOTE: We do not collect the "kernel_arg_name" metadata, because
    // the kernel arguments receive new names in the fused kernel.
    static SmallVector<StringRef> Keys{
        {"kernel_arg_addr_space", "kernel_arg_access_qual", "kernel_arg_type",
         "kernel_arg_base_type", "kernel_arg_type_qual"}};
    return Keys;
  }

  void postProcessKernel(Function *KernelFunc) const override {
    // Attach the kernel_arg_name metadata.
    SmallVector<Metadata *> KernelArgNames;
    for (auto &P : KernelFunc->args()) {
      KernelArgNames.push_back(
          MDString::get(LLVMMod->getContext(), P.getName()));
    }
    auto *ArgNameMD = MDTuple::get(LLVMMod->getContext(), KernelArgNames);
    KernelFunc->setMetadata("kernel_arg_name", ArgNameMD);

    static constexpr auto ITTStartWrapper = "__itt_offload_wi_start_wrapper";
    static constexpr auto ITTFinishWrapper = "__itt_offload_wi_finish_wrapper";
    // Remove all existing calls of the ITT instrumentation functions. Insert
    // new ones in the entry block of the fused kernel and every exit block if
    // the functions are present in the module. We cannot use the existing
    // SPIRITTAnnotations pass, because that pass might insert calls to
    // functions not present in the module (e.g., ITT instrumentations for
    // barriers). As the JITed module is not linked with libdevice anymore, the
    // functions would remain unresolved and cause the driver to fail.
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

  void createBarrierCall(IRBuilderBase &Builder,
                         int BarrierFlags) const override {
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

  // Corresponds to definition of spir_private and spir_local in
  // "clang/lib/Basic/Target/SPIR.h", "SPIRDefIsGenMap".
  unsigned getPrivateAddressSpace() const override { return 0; }
  unsigned getLocalAddressSpace() const override { return 3; }

  void updateAddressSpaceMetadata(Function *KernelFunc,
                                  ArrayRef<size_t> LocalSize,
                                  unsigned AddressSpace) const override {
    static constexpr unsigned AddressSpaceBitWidth{32};
    static constexpr StringLiteral KernelArgAddrSpaceMD{
        "kernel_arg_addr_space"};

    auto *NewAddrspace = ConstantAsMetadata::get(ConstantInt::get(
        IntegerType::get(LLVMMod->getContext(), AddressSpaceBitWidth),
        AddressSpace));
    if (auto *AddrspaceMD = dyn_cast_or_null<MDNode>(
            KernelFunc->getMetadata(KernelArgAddrSpaceMD))) {
      // If we have kernel_arg_addr_space metadata in the original function,
      // we should update it in the new one.
      SmallVector<Metadata *> NewInfo{AddrspaceMD->op_begin(),
                                      AddrspaceMD->op_end()};
      for (auto I : enumerate(LocalSize)) {
        if (I.value() == 0) {
          continue;
        }
        const auto Index = I.index();
        if (const auto *PtrTy =
                dyn_cast<PointerType>(KernelFunc->getArg(Index)->getType())) {
          if (PtrTy->getAddressSpace() == getLocalAddressSpace()) {
            NewInfo[Index] = NewAddrspace;
          }
        }
      }
      KernelFunc->setMetadata(KernelArgAddrSpaceMD,
                              MDNode::get(KernelFunc->getContext(), NewInfo));
    }
  }

  std::optional<BuiltinKind> getBuiltinKind(Function *F) const override {
    constexpr size_t NumBuiltinsToRemap{7};
    // clang-format off
    // Note: This list is sorted by the builtin name.
    constexpr std::array<std::pair<StringLiteral, BuiltinKind>,
                         NumBuiltinsToRemap>
        Map{{{"_Z25__spirv_BuiltInGlobalSizei",
              BuiltinKind::GlobalSizeRemapper},
             {"_Z26__spirv_BuiltInWorkgroupIdi",
              BuiltinKind::GroupIDRemapper},
             {"_Z27__spirv_BuiltInGlobalOffseti",
              BuiltinKind::GlobalOffsetRemapper},
             {"_Z28__spirv_BuiltInNumWorkgroupsi",
              BuiltinKind::NumWorkGroupsRemapper},
             {"_Z28__spirv_BuiltInWorkgroupSizei",
              BuiltinKind::LocalSizeRemapper},
             {"_Z32__spirv_BuiltInLocalInvocationIdi",
              BuiltinKind::LocalIDRemapper},
             {"_Z33__spirv_BuiltInGlobalInvocationIdi",
              BuiltinKind::GlobalIDRemapper}}};
    // clang-format on
    return mapArrayLookupValue(Map.begin(), Map.end(), F->getName());
  }

  bool shouldRemap(BuiltinKind K, const NDRange &SrcNDRange,
                   const NDRange &FusedNDRange) const override {
    switch (K) {
    case BuiltinKind::GlobalSizeRemapper:
    case BuiltinKind::GlobalOffsetRemapper:
      return true;
    case BuiltinKind::NumWorkGroupsRemapper:
    case BuiltinKind::LocalSizeRemapper:
      // Do not remap when the local size is not specified.
      return SrcNDRange.hasSpecificLocalSize();
    case BuiltinKind::GlobalIDRemapper:
    case BuiltinKind::LocalIDRemapper:
    case BuiltinKind::GroupIDRemapper:
      return requireIDRemapping(SrcNDRange, FusedNDRange);
    }
    llvm_unreachable("Unhandled kind");
  }

  bool isSafeToNotRemapBuiltin(Function *F) const override {
    constexpr std::size_t NumUnsafeBuiltins{8};
    // SPIRV builtins with kernel capabilities in alphabetical order.
    //
    // These builtins might need remapping, but are not supported by the
    // remapper, so we should abort kernel fusion if we find them during
    // remapping.
    constexpr std::array<StringLiteral, NumUnsafeBuiltins> UnsafeBuiltIns{
        "EnqueuedWorkgroupSize",
        "NumEnqueuedSubgroups",
        "NumSubgroups",
        "SubgroupId",
        "SubgroupLocalInvocationId",
        "SubgroupMaxSize",
        "SubgroupSize",
        "WorkDim"};
    constexpr StringLiteral SPIRVBuiltinNamespace{"spirv"};
    constexpr StringLiteral SPIRVBuiltinPrefix{"BuiltIn"};

    auto Name = F->getName();
    if (!(Name.contains(SPIRVBuiltinNamespace) &&
          Name.contains(SPIRVBuiltinPrefix))) {
      return true;
    }
    // Drop "spirv" namespace name and "BuiltIn" prefix.
    Name = Name.drop_front(Name.find(SPIRVBuiltinPrefix) +
                           SPIRVBuiltinPrefix.size());
    // Check that Name does not start with any name in UnsafeBuiltIns
    const auto *Iter =
        std::upper_bound(UnsafeBuiltIns.begin(), UnsafeBuiltIns.end(), Name);
    return Iter == UnsafeBuiltIns.begin() || !Name.starts_with(*(Iter - 1));
  }

  unsigned getIndexSpaceBuiltinBitwidth() const override { return 64; }

  void setMetadataForGeneratedFunction(Function *F) const override {
    auto &Ctx = F->getContext();
    F->setAttributes(AttributeList::get(
        Ctx,
        AttributeSet::get(
            Ctx, {Attribute::get(Ctx, Attribute::AttrKind::NoUnwind),
                  Attribute::get(Ctx, Attribute::AttrKind::AlwaysInline)}),
        {}, {}));
    F->setCallingConv(CallingConv::SPIR_FUNC);
  }

  Value *getGlobalIDWithoutOffset(IRBuilderBase &Builder,
                                  const NDRange &FusedNDRange,
                                  uint32_t Idx) const override {
    // The SPIR-V target only remaps IDs (and thus queries this method) if no
    // global offset is given.
    assert(FusedNDRange.getOffset() == NDRange::AllZeros);

    constexpr StringLiteral GetGlobalIDName{
        "_Z33__spirv_BuiltInGlobalInvocationIdi"};
    auto *M = Builder.GetInsertBlock()->getParent()->getParent();
    auto *F = M->getFunction(GetGlobalIDName);
    if (!F) {
      auto &Ctx = Builder.getContext();
      auto *Ty = FunctionType::get(Builder.getInt64Ty(), {Builder.getInt32Ty()},
                                   /*isVarArg*/ false);
      F = Function::Create(Ty, Function::ExternalLinkage, GetGlobalIDName, M);
      F->setAttributes(AttributeList::get(
          Ctx,
          AttributeSet::get(
              Ctx, {Attribute::get(Ctx, Attribute::AttrKind::WillReturn),
                    Attribute::get(Ctx, Attribute::AttrKind::NoUnwind)}),
          {}, {}));
      F->setCallingConv(CallingConv::SPIR_FUNC);
    }

    auto *Call = Builder.CreateCall(F, {Builder.getInt32(Idx)});
    Call->setAttributes(F->getAttributes());
    Call->setCallingConv(F->getCallingConv());

    return Call;
  }

  Function *createRemapperFunction(
      const Remapper &R, BuiltinKind K, StringRef OrigName, StringRef Name,
      Module *M, const jit_compiler::NDRange &SrcNDRange,
      const jit_compiler::NDRange &FusedNDRange) const override {
    auto &Ctx = M->getContext();
    IRBuilder<> Builder(Ctx);

    auto *Ty = FunctionType::get(Builder.getInt64Ty(), {Builder.getInt32Ty()},
                                 /*isVarArg*/ false);
    auto *F = Function::Create(Ty, Function::InternalLinkage, Name, *M);
    setMetadataForGeneratedFunction(F);

    auto *EntryBlock = BasicBlock::Create(Ctx, "entry", F);
    Builder.SetInsertPoint(EntryBlock);

    constexpr unsigned SYCLDimensions{3};
    // Vector holding all the possible values
    auto *Vec = cast<Value>(
        ConstantVector::getSplat(ElementCount::getFixed(SYCLDimensions),
                                 Builder.getInt64(R.getDefaultValue(K))));

    const auto NumDimensions =
        static_cast<uint32_t>(SrcNDRange.getDimensions());
    for (uint32_t I = 0; I < NumDimensions; ++I) {
      // Initialize vector
      Vec = Builder.CreateInsertElement(
          Vec, R.remap(K, Builder, SrcNDRange, FusedNDRange, I),
          Builder.getInt32(I));
    }

    // Get queried value
    Builder.CreateRet(Builder.CreateExtractElement(Vec, F->getArg(0)));

    return F;
  }
};

//
// NVPTXTargetFusionInfo
//
#ifdef FUSION_JIT_SUPPORT_PTX
class NVPTXTargetFusionInfo : public TargetFusionInfoImpl {
public:
  using TargetFusionInfoImpl::TargetFusionInfoImpl;

  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const override {
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
    auto *NewAnnotations =
        LLVMMod->getOrInsertNamedMetadata("nvvm.annotations");
    for (auto *Kernel : ValidKernels) {
      NewAnnotations->addOperand(Kernel);
    }
  }

  void addKernelFunction(Function *KernelFunc) const override {
    auto *NVVMAnnotations =
        LLVMMod->getOrInsertNamedMetadata("nvvm.annotations");
    auto *MDOne = ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(LLVMMod->getContext()), 1));
    auto *MDKernelString = MDString::get(LLVMMod->getContext(), "kernel");
    auto *MDFunc = ConstantAsMetadata::get(KernelFunc);
    SmallVector<Metadata *, 3> KernelMD({MDFunc, MDKernelString, MDOne});
    auto *Tuple = MDTuple::get(LLVMMod->getContext(), KernelMD);
    NVVMAnnotations->addOperand(Tuple);
  }

  ArrayRef<StringRef> getKernelMetadataKeys() const override {
    // FIXME: Check whether we need to take care of sycl_fixed_targets.
    static SmallVector<StringRef> Keys{{"kernel_arg_buffer_location",
                                        "kernel_arg_runtime_aligned",
                                        "kernel_arg_exclusive_ptr"}};
    return Keys;
  }

  ArrayRef<StringRef> getUniformKernelAttributes() const override {
    static SmallVector<StringRef> Keys{
        {"target-cpu", "target-features", "uniform-work-group-size"}};
    return Keys;
  }

  void createBarrierCall(IRBuilderBase &Builder,
                         int BarrierFlags) const override {
    if (BarrierFlags == -1) {
      return;
    }
    // Emit a call to llvm.nvvm.barrier0. From the user manual of the NVPTX
    // backend: "The ‘@llvm.nvvm.barrier0()’ intrinsic emits a PTX bar.sync 0
    // instruction, equivalent to the __syncthreads() call in CUDA."
    Builder.CreateIntrinsic(Intrinsic::NVVMIntrinsics::nvvm_barrier0, {}, {});
  }

  // Corresponds to the definitions in the LLVM NVPTX backend user guide:
  // https://llvm.org/docs/NVPTXUsage.html#address-spaces
  unsigned getPrivateAddressSpace() const override { return 0; }
  unsigned getLocalAddressSpace() const override { return 3; }

  std::optional<BuiltinKind> getBuiltinKind(Function *F) const override {
    // PTX doesn't have intrinsics for global sizes and global IDs:
    // https://www.llvm.org/docs/NVPTXUsage.html#reading-ptx-special-registers

    if (!F->isIntrinsic())
      return {};
    switch (F->getIntrinsicID()) {
    case Intrinsic::nvvm_implicit_offset:
      return BuiltinKind::GlobalOffsetRemapper;
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    case Intrinsic::nvvm_read_ptx_sreg_tid_z:
      return BuiltinKind::LocalIDRemapper;
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
      return BuiltinKind::GroupIDRemapper;
    case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
    case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
    case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
      return BuiltinKind::LocalSizeRemapper;
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
      return BuiltinKind::NumWorkGroupsRemapper;
    default:
      return {};
    }
  }

  bool shouldRemap(BuiltinKind K, const NDRange &SrcNDRange,
                   const NDRange &FusedNDRange) const override {
    switch (K) {
    case BuiltinKind::LocalSizeRemapper:
    case BuiltinKind::NumWorkGroupsRemapper:
    case BuiltinKind::GlobalOffsetRemapper:
      return true;
    case BuiltinKind::LocalIDRemapper:
    case BuiltinKind::GroupIDRemapper:
      // If the local size is not specified, we have to unconditionally remap
      // local and group IDs to guarantee correct global IDs.
      return !SrcNDRange.hasSpecificLocalSize() ||
             requireIDRemapping(SrcNDRange, FusedNDRange);
    default:
      llvm_unreachable("Unhandled kind");
    }
  }

  bool isSafeToNotRemapBuiltin(Function *F) const override {
    // `SubgroupLocalInvocationId` lowers to the `laneid`.
    // Other subgroup-related builtins are computed from standard getters
    // (workgroup size, local ID etc.) and constants (subgroup max size := 32),
    // so we can't filter them out here.
    return F->getIntrinsicID() != Intrinsic::nvvm_read_ptx_sreg_laneid;
  }

  unsigned getIndexSpaceBuiltinBitwidth() const override { return 32; }

  void setMetadataForGeneratedFunction(Function *F) const override {
    auto &Ctx = F->getContext();
    if (F->getName().contains("__global_offset_remapper")) {
      F->setAttributes(AttributeList::get(
          Ctx,
          AttributeSet::get(
              Ctx, {Attribute::get(Ctx, Attribute::AttrKind::NoUnwind),
                    Attribute::get(Ctx, Attribute::AttrKind::Speculatable),

                    Attribute::get(Ctx, Attribute::AttrKind::AlwaysInline)}),
          {}, {}));
    } else {
      F->setAttributes(AttributeList::get(
          Ctx,
          AttributeSet::get(
              Ctx, {Attribute::get(Ctx, Attribute::AttrKind::NoCallback),
                    Attribute::get(Ctx, Attribute::AttrKind::NoFree),
                    Attribute::get(Ctx, Attribute::AttrKind::NoSync),
                    Attribute::get(Ctx, Attribute::AttrKind::NoUnwind),
                    Attribute::get(Ctx, Attribute::AttrKind::Speculatable),
                    Attribute::get(Ctx, Attribute::AttrKind::WillReturn),
                    Attribute::get(Ctx, Attribute::AttrKind::AlwaysInline)}),
          {}, {}));
    }
    F->setMemoryEffects(MemoryEffects::none());
  }

  Value *getGlobalIDWithoutOffset(IRBuilderBase &Builder,
                                  const NDRange &FusedNDRange,
                                  uint32_t Idx) const override {
    // Construct (or reuse) a helper function to query the global ID.
    std::string GetGlobalIDName =
        ("__global_id_" + Twine(static_cast<char>('x' + Idx))).str();
    auto *M = Builder.GetInsertBlock()->getParent()->getParent();
    auto *F = M->getFunction(GetGlobalIDName);
    if (!F) {
      auto IP = Builder.saveIP();
      auto *I32Ty = Builder.getInt32Ty();
      auto *Ty = FunctionType::get(I32Ty,
                                   /*isVarArg*/ false);
      F = Function::Create(Ty, Function::InternalLinkage, GetGlobalIDName, M);
      setMetadataForGeneratedFunction(F);

      auto *EntryBlock = BasicBlock::Create(Builder.getContext(), "entry", F);
      Builder.SetInsertPoint(EntryBlock);

      // Compute `global_id.i = group_id.i * local_size.i + local_id.i`.
      auto *WorkGroupId = Builder.CreateIntrinsic(
          I32Ty, Intrinsic::nvvm_read_ptx_sreg_ctaid_x + Idx, {});
      auto *LocalSize = Builder.CreateIntrinsic(
          I32Ty, Intrinsic::nvvm_read_ptx_sreg_ntid_x + Idx, {});
      auto *LocalId = Builder.CreateIntrinsic(
          I32Ty, Intrinsic::nvvm_read_ptx_sreg_tid_x + Idx, {});
      Builder.CreateRet(Builder.CreateAdd(
          Builder.CreateMul(WorkGroupId, LocalSize), LocalId));
      Builder.restoreIP(IP);
    }

    auto *Call = Builder.CreateCall(F);
    Call->setAttributes(F->getAttributes());
    Call->setCallingConv(F->getCallingConv());

    return Call;
  }

  Function *createRemapperFunction(const Remapper &R, BuiltinKind K,
                                   StringRef OrigName, StringRef Name,
                                   Module *M, const NDRange &SrcNDRange,
                                   const NDRange &FusedNDRange) const override {
    auto &Ctx = M->getContext();
    IRBuilder<> Builder(Ctx);

    // Handle global offset first because its return type is different from the
    // other index space getters.
    if (K == BuiltinKind::GlobalOffsetRemapper) {
      // Collect offset value for all dimensions.
      const auto NumDimensions =
          static_cast<uint32_t>(SrcNDRange.getDimensions());
      SmallVector<Constant *, 3> Offsets;
      for (uint32_t I = 0; I < 3; ++I) {
        if (I < NumDimensions) {
          Value *Remapped = R.remap(K, Builder, SrcNDRange, FusedNDRange, I);
          auto *RemappedC = dyn_cast<Constant>(Remapped);
          assert(RemappedC && "Global offset is not constant");
          Offsets.push_back(RemappedC);
        } else {
          Offsets.push_back(Builder.getInt32(R.getDefaultValue(K)));
        }
      }

      // `llvm.nvvm.implicit.offset` returns a pointer to a 3-element array.
      // https://github.com/intel/llvm/blob/sycl/sycl/doc/design/CompilerAndRuntimeDesign.md#global-offset-support
      //
      // Hence, create a new global constant initialized to the offsets
      // collected above and return its address.
      auto *CTy = ArrayType::get(Builder.getInt32Ty(), 3);
      auto *C = ConstantArray::get(CTy, Offsets);
      auto *GV =
          new GlobalVariable(*M, CTy, /*isConstant*/ true,
                             GlobalValue::InternalLinkage, C, Name + "__const");
      auto *FTy = FunctionType::get(Builder.getPtrTy(), /*isVarArg*/ false);
      auto *F = Function::Create(FTy, Function::InternalLinkage, Name, *M);
      setMetadataForGeneratedFunction(F);

      auto *EntryBlock = BasicBlock::Create(Ctx, "entry", F);
      Builder.SetInsertPoint(EntryBlock);
      Builder.CreateRet(GV);

      return F;
    }

    // All other index space getters are `() -> i32`.
    // https://www.llvm.org/docs/NVPTXUsage.html#llvm-nvvm-read-ptx-sreg
    // As these don't take a dimension index argument, we have to do some
    // trickery to map the x/y/z suffixes to indices, and vice versa.
    auto WrapValInFunc =
        [&](std::function<Value *(uint32_t)> ValueFn) -> Function * {
      auto Suffix = OrigName.take_back();
      assert(Suffix[0] >= 'x' && Suffix[0] <= 'z');
      uint32_t Idx = Suffix[0] - 'x';

      std::string RemapperName =
          (Name + Twine('_') + Twine(static_cast<char>('x' + Idx))).str();

      auto *FTy = FunctionType::get(Builder.getInt32Ty(), /*isVarArg*/ false);
      auto *F =
          Function::Create(FTy, Function::InternalLinkage, RemapperName, *M);
      setMetadataForGeneratedFunction(F);

      auto *EntryBlock = BasicBlock::Create(Ctx, "entry", F);
      Builder.SetInsertPoint(EntryBlock);
      auto *Res = ValueFn(Idx);
      Builder.CreateRet(Res);

      return F;
    };

    // If the kernel was launched with an `nd_range`, the local size is known
    // and guaranteed to divide the global size. Hence, we can remap all
    // intrinsics in the target-independent way.
    if (SrcNDRange.hasSpecificLocalSize()) {
      return WrapValInFunc([&](uint32_t Idx) {
        return R.remap(K, Builder, SrcNDRange, FusedNDRange, Idx);
      });
    }

    // Otherwise, we need to remap the intrinsics in a way that is compatible
    // with the lowered computation of the global sizes and global IDs:
    //
    // ```
    // global_size.i = num_work_groups.i * local_size.i
    // global_id.i   = group_id.i * local_size.i + local_id.i + global_offset[i]
    // ```
    //
    // We infer from the absence of the local size that the kernel was launched
    // on a regular range. This means that user code cannot query local sizes,
    // local IDs or workgroup IDs, and we're free to assume values that work in
    // context of the global size and global ID calcutations. To that end, we
    // create remappers to simulate a single workgroup with the global size of
    // the original range.
    //
    // ```
    // num_work_groups.i := 1
    // work_group_id.i   := 0
    // local_size.i      := original global_size
    // local_id.i        := remapped global_ID w/o global offset, computed via
    //                      global linear ID
    // ```
    switch (K) {
    case BuiltinKind::NumWorkGroupsRemapper:
    case BuiltinKind::GroupIDRemapper:
      return WrapValInFunc(
          [&](uint32_t Idx) { return Builder.getInt32(R.getDefaultValue(K)); });
    case BuiltinKind::LocalSizeRemapper:
      return WrapValInFunc([&](uint32_t Idx) {
        return R.remap(BuiltinKind::GlobalSizeRemapper, Builder, SrcNDRange,
                       FusedNDRange, Idx);
      });
    case BuiltinKind::LocalIDRemapper:
      return WrapValInFunc([&](uint32_t Idx) {
        return R.remap(BuiltinKind::GlobalIDRemapper, Builder, SrcNDRange,
                       FusedNDRange, Idx);
      });
    default:
      // Global offset was already handled above.
      llvm_unreachable("unhandled builtin");
    }
  }
};
#endif // FUSION_JIT_SUPPORT_PTX

} // anonymous namespace

//
// TargetFusionInfo
//

TargetFusionInfo::TargetFusionInfo(llvm::Module *Mod) {
  llvm::Triple Tri(Mod->getTargetTriple());
#ifdef FUSION_JIT_SUPPORT_PTX
  if (Tri.isNVPTX()) {
    Impl = std::make_shared<NVPTXTargetFusionInfo>(Mod);
    return;
  }
#endif // FUSION_JIT_SUPPORT_PTX
  if (Tri.isSPIRV() || Tri.isSPIR()) {
    Impl = std::make_shared<SPIRVTargetFusionInfo>(Mod);
    return;
  }
  llvm_unreachable("Unsupported target for fusion");
}

void TargetFusionInfo::notifyFunctionsDelete(
    llvm::ArrayRef<Function *> Funcs) const {
  Impl->notifyFunctionsDelete(Funcs);
}

void TargetFusionInfo::addKernelFunction(llvm::Function *KernelFunc) const {
  Impl->addKernelFunction(KernelFunc);
}

void TargetFusionInfo::postProcessKernel(Function *KernelFunc) const {
  Impl->postProcessKernel(KernelFunc);
}

llvm::ArrayRef<llvm::StringRef>
TargetFusionInfo::getKernelMetadataKeys() const {
  return Impl->getKernelMetadataKeys();
}

void TargetFusionInfo::createBarrierCall(IRBuilderBase &Builder,
                                         int BarrierFlags) const {
  Impl->createBarrierCall(Builder, BarrierFlags);
}

unsigned TargetFusionInfo::getPrivateAddressSpace() const {
  return Impl->getPrivateAddressSpace();
}

unsigned TargetFusionInfo::getLocalAddressSpace() const {
  return Impl->getLocalAddressSpace();
}

void TargetFusionInfo::updateAddressSpaceMetadata(Function *KernelFunc,
                                                  ArrayRef<size_t> LocalSize,
                                                  unsigned AddressSpace) const {
  Impl->updateAddressSpaceMetadata(KernelFunc, LocalSize, AddressSpace);
}

llvm::ArrayRef<llvm::StringRef>
TargetFusionInfo::getUniformKernelAttributes() const {
  return Impl->getUniformKernelAttributes();
}

std::optional<BuiltinKind> TargetFusionInfo::getBuiltinKind(Function *F) const {
  return Impl->getBuiltinKind(F);
}

bool TargetFusionInfo::shouldRemap(BuiltinKind K, const NDRange &SrcNDRange,
                                   const NDRange &FusedNDRange) const {
  return Impl->shouldRemap(K, SrcNDRange, FusedNDRange);
}

bool TargetFusionInfo::isSafeToNotRemapBuiltin(Function *F) const {
  return Impl->isSafeToNotRemapBuiltin(F);
}

unsigned TargetFusionInfo::getIndexSpaceBuiltinBitwidth() const {
  return Impl->getIndexSpaceBuiltinBitwidth();
}

void TargetFusionInfo::setMetadataForGeneratedFunction(Function *F) const {
  Impl->setMetadataForGeneratedFunction(F);
}

Value *TargetFusionInfo::getGlobalIDWithoutOffset(IRBuilderBase &Builder,
                                                  const NDRange &FusedNDRange,
                                                  uint32_t Idx) const {
  return Impl->getGlobalIDWithoutOffset(Builder, FusedNDRange, Idx);
}

Function *TargetFusionInfo::createRemapperFunction(
    const Remapper &R, BuiltinKind K, StringRef OrigName, StringRef Name,
    Module *M, const NDRange &SrcNDRange, const NDRange &FusedNDRange) const {
  return Impl->createRemapperFunction(R, K, OrigName, Name, M, SrcNDRange,
                                      FusedNDRange);
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

} // namespace llvm
