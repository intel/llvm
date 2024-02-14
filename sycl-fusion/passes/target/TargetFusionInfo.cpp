//==---------------------- TargetFusionInfo.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetFusionInfo.h"
#include "Kernel.h"

#include "Kernel.h"
#include "NDRangesHelper.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace jit_compiler;

extern llvm::cl::opt<bool> UseNewDbgInfoFormat;

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
                                 BarrierFlags BarrierFlags) const = 0;

  virtual unsigned getPrivateAddressSpace() const = 0;

  virtual unsigned getLocalAddressSpace() const = 0;

  virtual void
  updateAddressSpaceMetadata([[maybe_unused]] Function *KernelFunc,
                             [[maybe_unused]] ArrayRef<bool> ArgIsPromoted,
                             [[maybe_unused]] unsigned AddressSpace) const {}

  virtual std::optional<BuiltinKind> getBuiltinKind(Function *F) const = 0;

  virtual bool shouldRemap(BuiltinKind K, const NDRange &SrcNDRange,
                           const NDRange &FusedNDRange) const = 0;

  virtual Error
  scanForBuiltinsToRemap(Function *F, Remapper &R,
                         const jit_compiler::NDRange &SrcNDRange,
                         const jit_compiler::NDRange &FusedNDRange) const {
    for (auto &I : instructions(F)) {
      if (auto *Call = dyn_cast<CallBase>(&I)) {
        // Recursive call
        auto *OldF = Call->getCalledFunction();
        auto ErrOrNewF = R.remapBuiltins(OldF, SrcNDRange, FusedNDRange);
        if (auto Err = ErrOrNewF.takeError()) {
          return Err;
        }
        // Override called function.
        auto *NewF = *ErrOrNewF;
        Call->setCalledFunction(NewF);
        Call->setCallingConv(NewF->getCallingConv());
        Call->setAttributes(NewF->getAttributes());
      }
    }
    return Error::success();
  }

  virtual bool isSafeToNotRemapBuiltin(Function *F) const = 0;

  virtual unsigned getIndexSpaceBuiltinBitwidth() const = 0;

  virtual void setMetadataForGeneratedFunction(Function *F) const = 0;

  virtual Value *getGlobalIDWithoutOffset(IRBuilderBase &Builder,
                                          const NDRange &FusedNDRange,
                                          uint32_t Idx) const = 0;

  virtual Function *
  createRemapperFunction(const Remapper &R, BuiltinKind K, StringRef OrigName,
                         Module *M, const NDRange &SrcNDRange,
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
                         BarrierFlags BarrierFlags) const override {
    if (isNoBarrierFlag(BarrierFlags)) {
      return;
    }

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
      F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);
    }

    // See
    // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-
    SmallVector<Value *> Args{
        Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
        Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
        Builder.getInt32(0x10 |
                         (hasLocalBarrierFlag(BarrierFlags) ? 0x100 : 0x0) |
                         ((hasGlobalBarrierFlag(BarrierFlags) ? 0x200 : 0x0)))};

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
                                  ArrayRef<bool> ArgIsPromoted,
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
      for (auto I : enumerate(ArgIsPromoted)) {
        if (!I.value()) {
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
      F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);
    }

    auto *Call = Builder.CreateCall(F, {Builder.getInt32(Idx)});
    Call->setAttributes(F->getAttributes());
    Call->setCallingConv(F->getCallingConv());

    return Call;
  }

  Function *createRemapperFunction(
      const Remapper &R, BuiltinKind K, StringRef OrigName, Module *M,
      const jit_compiler::NDRange &SrcNDRange,
      const jit_compiler::NDRange &FusedNDRange) const override {
    const auto Name = Remapper::getFunctionName(K, SrcNDRange, FusedNDRange);
    assert(!M->getFunction(Name) && "Function name should be unique");
    auto &Ctx = M->getContext();
    IRBuilder<> Builder(Ctx);

    auto *Ty = FunctionType::get(Builder.getInt64Ty(), {Builder.getInt32Ty()},
                                 /*isVarArg*/ false);
    auto *F = Function::Create(Ty, Function::InternalLinkage, Name, *M);
    setMetadataForGeneratedFunction(F);
    F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);

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

class NVPTXAMDGCNTargetFusionInfoBase : public TargetFusionInfoImpl {
public:
  using TargetFusionInfoImpl::TargetFusionInfoImpl;

  void notifyFunctionsDelete(StringRef MDName,
                             llvm::ArrayRef<Function *> Funcs) const {
    SmallPtrSet<Constant *, 8> DeletedFuncs{Funcs.begin(), Funcs.end()};
    SmallVector<MDNode *> ValidKernels;
    auto *OldAnnotations = LLVMMod->getNamedMetadata(MDName);
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
    auto *NewAnnotations = LLVMMod->getOrInsertNamedMetadata(MDName);
    for (auto *Kernel : ValidKernels) {
      NewAnnotations->addOperand(Kernel);
    }
  }

  void addKernelFunction(StringRef MDName, Function *KernelFunc) const {
    auto *Annotations = LLVMMod->getOrInsertNamedMetadata(MDName);
    auto *MDOne = ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(LLVMMod->getContext()), 1));
    auto *MDKernelString = MDString::get(LLVMMod->getContext(), "kernel");
    auto *MDFunc = ConstantAsMetadata::get(KernelFunc);
    SmallVector<Metadata *, 3> KernelMD({MDFunc, MDKernelString, MDOne});
    auto *Tuple = MDTuple::get(LLVMMod->getContext(), KernelMD);
    Annotations->addOperand(Tuple);
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

  virtual std::array<Value *, 3> getLocalGridInfo(IRBuilderBase &Builder,
                                                  uint32_t Idx) const = 0;

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
      F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);

      auto *EntryBlock = BasicBlock::Create(Builder.getContext(), "entry", F);
      Builder.SetInsertPoint(EntryBlock);

      // Compute `global_id.i = group_id.i * local_size.i + local_id.i`.
      auto [WorkGroupId, LocalSize, LocalId] = getLocalGridInfo(Builder, Idx);

      Builder.CreateRet(Builder.CreateAdd(
          Builder.CreateMul(WorkGroupId, LocalSize), LocalId));
      Builder.restoreIP(IP);
    }

    auto *Call = Builder.CreateCall(F);
    Call->setAttributes(F->getAttributes());
    Call->setCallingConv(F->getCallingConv());

    return Call;
  }

  Function *createRemapperFunctionWithIdx(const Remapper &R, BuiltinKind K,
                                          uint32_t Idx, StringRef Name,
                                          Module *M, const NDRange &SrcNDRange,
                                          const NDRange &FusedNDRange) const {
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
      auto *GV = new GlobalVariable(
          *M, CTy, /*isConstant*/ true, GlobalValue::InternalLinkage, C,
          Name + "__const", /*InsertBefore*/ nullptr,
          GlobalVariable::NotThreadLocal, getPrivateAddressSpace());
      auto *FTy = FunctionType::get(Builder.getPtrTy(getPrivateAddressSpace()),
                                    /*isVarArg*/ false);
      auto *F = Function::Create(FTy, Function::InternalLinkage, Name, *M);
      setMetadataForGeneratedFunction(F);
      F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);

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
      auto *FTy = FunctionType::get(Builder.getInt32Ty(), /*isVarArg*/ false);
      auto *F = Function::Create(FTy, Function::InternalLinkage, Name, *M);
      setMetadataForGeneratedFunction(F);
      F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);

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
    case BuiltinKind::GlobalSizeRemapper: /* only AMDGCN */
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

//
// NVPTXTargetFusionInfo
//
#ifdef FUSION_JIT_SUPPORT_PTX
class NVPTXTargetFusionInfo final : public NVPTXAMDGCNTargetFusionInfoBase {
public:
  using NVPTXAMDGCNTargetFusionInfoBase::NVPTXAMDGCNTargetFusionInfoBase;

  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const override {
    NVPTXAMDGCNTargetFusionInfoBase::notifyFunctionsDelete("nvvm.annotations",
                                                           Funcs);
  }

  void addKernelFunction(Function *KernelFunc) const override {
    NVPTXAMDGCNTargetFusionInfoBase::addKernelFunction("nvvm.annotations",
                                                       KernelFunc);
  }

  void createBarrierCall(IRBuilderBase &Builder,
                         BarrierFlags BarrierFlags) const override {
    if (isNoBarrierFlag(BarrierFlags)) {
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

  bool isSafeToNotRemapBuiltin(Function *F) const override {
    // `SubgroupLocalInvocationId` lowers to the `laneid`.
    // Other subgroup-related builtins are computed from standard getters
    // (workgroup size, local ID etc.) and constants (subgroup max size := 32),
    // so we can't filter them out here.
    return F->getIntrinsicID() != Intrinsic::nvvm_read_ptx_sreg_laneid;
  }

  std::array<Value *, 3> getLocalGridInfo(IRBuilderBase &Builder,
                                          uint32_t Idx) const override {
    auto *I32Ty = Builder.getInt32Ty();
    auto *WorkGroupId = Builder.CreateIntrinsic(
        I32Ty, Intrinsic::nvvm_read_ptx_sreg_ctaid_x + Idx, {});
    auto *LocalSize = Builder.CreateIntrinsic(
        I32Ty, Intrinsic::nvvm_read_ptx_sreg_ntid_x + Idx, {});
    auto *LocalId = Builder.CreateIntrinsic(
        I32Ty, Intrinsic::nvvm_read_ptx_sreg_tid_x + Idx, {});
    return {WorkGroupId, LocalSize, LocalId};
  }

  Function *createRemapperFunction(const Remapper &R, BuiltinKind K,
                                   StringRef OrigName, Module *M,
                                   const NDRange &SrcNDRange,
                                   const NDRange &FusedNDRange) const override {
    auto &Ctx = M->getContext();
    IRBuilder<> Builder(Ctx);
    uint32_t Idx = -1;

    // Handle global offset first because its return type is different from the
    // other index space getters.
    if (K != BuiltinKind::GlobalOffsetRemapper) {
      auto Suffix = OrigName.take_back();
      assert(Suffix[0] >= 'x' && Suffix[0] <= 'z');
      Idx = Suffix[0] - 'x';
    }
    const std::string Name =
        Remapper::getFunctionName(K, SrcNDRange, FusedNDRange, Idx);
    assert(!M->getFunction(Name) && "Function name should be unique");

    return createRemapperFunctionWithIdx(R, K, Idx, Name, M, SrcNDRange,
                                         FusedNDRange);
  }
};
#endif // FUSION_JIT_SUPPORT_PTX

//
// AMDGCNTargetFusionInfo
//
#ifdef FUSION_JIT_SUPPORT_AMDGCN
class AMDGCNTargetFusionInfo final : public NVPTXAMDGCNTargetFusionInfoBase {
  using Base = NVPTXAMDGCNTargetFusionInfoBase;

public:
  using NVPTXAMDGCNTargetFusionInfoBase::NVPTXAMDGCNTargetFusionInfoBase;

  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const override {
    NVPTXAMDGCNTargetFusionInfoBase::notifyFunctionsDelete("amdgcn.annotations",
                                                           Funcs);
  }

  void addKernelFunction(Function *KernelFunc) const override {
    KernelFunc->setCallingConv(CallingConv::AMDGPU_KERNEL);
    NVPTXAMDGCNTargetFusionInfoBase::addKernelFunction("amdgcn.annotations",
                                                       KernelFunc);
  }

  void createBarrierCall(IRBuilderBase &Builder,
                         BarrierFlags BarrierFlags) const override {
    if (isNoBarrierFlag(BarrierFlags)) {
      return;
    }
    // Following implemention in
    // libclc/amdgcn-amdhsa/libspirv/synchronization/barrier.cl
    llvm::AtomicOrdering AO = llvm::AtomicOrdering::SequentiallyConsistent;
    llvm::SyncScope::ID SSID =
        LLVMMod->getContext().getOrInsertSyncScopeID("workgroup");
    Builder.CreateFence(AO, SSID);
    Builder.CreateIntrinsic(Intrinsic::AMDGCNIntrinsics::amdgcn_s_barrier, {},
                            {});
  }

  // Corresponds to the definitions in the LLVM AMDGCN backend user guide:
  // https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces
  unsigned getPrivateAddressSpace() const override { return 5; }
  unsigned getLocalAddressSpace() const override { return 3; }
  unsigned getConstantAddressSpace() const { return 4; }

  std::optional<BuiltinKind> getBuiltinKind(Function *F) const override {
    if (!F->isIntrinsic())
      return {};
    switch (F->getIntrinsicID()) {
    case Intrinsic::amdgcn_implicit_offset:
      return BuiltinKind::GlobalOffsetRemapper;
    case Intrinsic::amdgcn_workitem_id_x:
    case Intrinsic::amdgcn_workitem_id_y:
    case Intrinsic::amdgcn_workitem_id_z:
      return BuiltinKind::LocalIDRemapper;
    case Intrinsic::amdgcn_workgroup_id_x:
    case Intrinsic::amdgcn_workgroup_id_y:
    case Intrinsic::amdgcn_workgroup_id_z:
      return BuiltinKind::GroupIDRemapper;
    case Intrinsic::amdgcn_dispatch_ptr:
    case Intrinsic::amdgcn_implicitarg_ptr:
      llvm_unreachable("amdgcn_dispatch_ptr and amdgcn_implicitarg_ptr "
                       "requires complex mapping");
    default:
      return {};
    }
  }

  bool isSafeToNotRemapBuiltin(Function *F) const override {
    // `SubgroupLocalInvocationId` lowers to the `mbcnt`.
    // Other subgroup-related builtins are computed from standard getters
    // (workgroup size, local ID etc.) and constants (subgroup max size := 32),
    // so we can't filter them out here.
    switch (F->getIntrinsicID()) {
    case Intrinsic::amdgcn_mbcnt_hi:
    case Intrinsic::amdgcn_mbcnt_lo:
      return false;
    default:
      return true;
    }
  }

  std::array<Value *, 3> getLocalGridInfo(IRBuilderBase &Builder,
                                          uint32_t Idx) const override {
    constexpr auto LocalSizeXOffset = 2;

    auto *I16Ty = Builder.getInt16Ty();
    auto *I32Ty = Builder.getInt32Ty();
    auto *CASPtrTy = Builder.getPtrTy(getConstantAddressSpace());

    // The backend provides intrinsics for getting the IDs, ...
    auto *WorkGroupId = Builder.CreateIntrinsic(
        I32Ty, Intrinsic::amdgcn_workgroup_id_x + Idx, {});
    auto *LocalId = Builder.CreateIntrinsic(
        I32Ty, Intrinsic::amdgcn_workitem_id_x + Idx, {});

    // ... but the local size must be queried via the dispatch pointer.
    auto *DPtr =
        Builder.CreateIntrinsic(CASPtrTy, Intrinsic::amdgcn_dispatch_ptr, {});
    auto *GEP = Builder.CreateInBoundsGEP(
        I16Ty, DPtr, Builder.getInt64(LocalSizeXOffset + Idx));
    auto *LSLoad = Builder.CreateLoad(I16Ty, GEP);
    auto *LocalSize = Builder.CreateZExt(LSLoad, I32Ty);

    return {WorkGroupId, LocalSize, LocalId};
  }

  Function *createRemapperFunction(const Remapper &R, BuiltinKind K,
                                   StringRef OrigName, Module *M,
                                   const NDRange &SrcNDRange,
                                   const NDRange &FusedNDRange) const override {
    auto &Ctx = M->getContext();
    IRBuilder<> Builder(Ctx);
    uint32_t Idx = 0;

    // Handle global offset first because its return type is different from the
    // other index space getters.
    if (K != BuiltinKind::GlobalOffsetRemapper) {
      auto Suffix = OrigName.take_back();
      assert(Suffix[0] >= 'x' && Suffix[0] <= 'z');
      Idx = Suffix[0] - 'x';
    }
    const std::string Name =
        Remapper::getFunctionName(K, SrcNDRange, FusedNDRange, Idx);
    assert(!M->getFunction(Name) && "Function name should be unique");

    return createRemapperFunctionWithIdx(R, K, Idx, Name, M, SrcNDRange,
                                         FusedNDRange);
  }

  Error collectDispatchedId(
      Instruction *Call,
      llvm::SmallMapVector<Instruction *, std::pair<BuiltinKind, uint32_t>, 16>
          &IdxAccess) const {
    // Local- and global size can only be queried via the dispatch pointer. This
    // method scans the users of a call to the dispatch pointer intrinsic for
    // GEPs and subsequent loads.
    //
    // Unfortunately, the offsets are not documented in LLVM backend guide; the
    // "best" reference is the libclc implementation in
    // `libclc/amdgcn-amdhsa/libspirv/workitem/get_local_size.cl` and
    // `get_global_size.cl`.

    llvm::SmallVector<std::pair<Instruction *, APInt>, 32> OffsetTracker;
    const DataLayout &DL = Call->getModule()->getDataLayout();
    const unsigned IndexSizeInBits =
        DL.getIndexSizeInBits(getConstantAddressSpace());

    OffsetTracker.push_back(
        {Call, APInt{IndexSizeInBits, 0, /*isSigned=*/true}});

    while (!OffsetTracker.empty()) {
      auto [I, Offset] = OffsetTracker.pop_back_val();
      for (User *U : I->users()) {
        Instruction *InsnUser = dyn_cast<Instruction>(U);
        if (!InsnUser)
          continue;

        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
          APInt GEPOffset{IndexSizeInBits, 0, /*isSigned=*/true};
          if (GEP->accumulateConstantOffset(DL, GEPOffset)) {
            OffsetTracker.push_back({GEP, Offset + GEPOffset});
            continue;
          }
          // Non-constant offset; give up.
          return createStringError(inconvertibleErrorCode(),
                                   "Cannot track dispatch ptr use");
        }

        if (LoadInst *Load = dyn_cast<LoadInst>(U)) {
          BuiltinKind BK;
          uint32_t Dim = 0;
          int64_t OffsetInt = Offset.getSExtValue();
          switch (OffsetInt) {
          case 4:
            BK = BuiltinKind::LocalSizeRemapper;
            Dim = 0;
            break;
          case 6:
            BK = BuiltinKind::LocalSizeRemapper;
            Dim = 1;
            break;
          case 8:
            BK = BuiltinKind::LocalSizeRemapper;
            Dim = 2;
            break;
          case 12:
            BK = BuiltinKind::GlobalSizeRemapper;
            Dim = 0;
            break;
          case 16:
            BK = BuiltinKind::GlobalSizeRemapper;
            Dim = 1;
            break;
          case 20:
            BK = BuiltinKind::GlobalSizeRemapper;
            Dim = 2;
            break;
          default:
            if (OffsetInt >= 4 && OffsetInt <= 23) {
              return createStringError(inconvertibleErrorCode(),
                                       "Internal error, invalid offset");
            }
            // The dispatch pointer has other legitimate uses, no remapping
            // needed in those cases.
            continue;
          }

          IdxAccess.insert({Load, {BK, Dim}});
          continue;
        }

        // Unexpected user; give up.
        return createStringError(inconvertibleErrorCode(),
                                 "Cannot track dispatch ptr use");
      }
    }

    return Error::success();
  }

  Error scanForBuiltinsToRemap(
      Function *F, Remapper &R, const jit_compiler::NDRange &SrcNDRange,
      const jit_compiler::NDRange &FusedNDRange) const override {
    llvm::SmallMapVector<Instruction *, std::pair<BuiltinKind, uint32_t>, 16>
        IdxAccess;

    // Scan for calls, recursively remap (simple) built-ins, and populate the
    // `IdxAccess` datastructure to capture loads from the dispatch pointer.
    for (auto &I : instructions(F)) {
      if (auto *Call = dyn_cast<CallBase>(&I)) {
        // Recursive call
        auto *OldF = Call->getCalledFunction();
        if (OldF->getIntrinsicID() == Intrinsic::amdgcn_dispatch_ptr) {
          if (auto Err = collectDispatchedId(Call, IdxAccess)) {
            return Err;
          }
          continue;
        }
        auto ErrOrNewF = R.remapBuiltins(OldF, SrcNDRange, FusedNDRange);
        if (auto Err = ErrOrNewF.takeError()) {
          return Err;
        }
        // Override called function.
        auto *NewF = *ErrOrNewF;
        Call->setCalledFunction(NewF);
        Call->setCallingConv(NewF->getCallingConv());
        Call->setAttributes(NewF->getAttributes());
      }
    }

    // Replace loads representing a builtin that needs to be remapped.
    llvm::SmallDenseMap<
        std::tuple<uint8_t, uint32_t, const NDRange *, const NDRange *>,
        Function *>
        DispatchRemap;
    Module *M = F->getParent();
    for (auto &Elt : IdxAccess) {
      auto *V = Elt.first;
      auto [BK, Idx] = Elt.second;
      Function *&Cache = DispatchRemap[decltype(DispatchRemap)::key_type{
          static_cast<uint8_t>(BK), Idx, &SrcNDRange, &FusedNDRange}];
      if (!Cache) {
        const auto Name =
            Remapper::getFunctionName(BK, SrcNDRange, FusedNDRange, Idx);

        if (!(Cache = M->getFunction(Name))) {
          Cache = createRemapperFunctionWithIdx(
              R, BK, Idx, Name, F->getParent(), SrcNDRange, FusedNDRange);
        }
      }
      IRBuilder<> Builder(V);
      auto *Call = Builder.CreateCall(Cache);
      Call->setAttributes(Cache->getAttributes());
      Call->setCallingConv(Cache->getCallingConv());
      // Truncation is required for the local size load, which is only `i16`.
      // The cast should be optimized away after inlining the remapper.
      auto *Cast = Builder.CreateTrunc(Call, V->getType());
      V->replaceAllUsesWith(Cast);
    }

    return Error::success();
  }
};
#endif // FUSION_JIT_SUPPORT_ADMGCN

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
#ifdef FUSION_JIT_SUPPORT_AMDGCN
  if (Tri.isAMDGCN()) {
    Impl = std::make_shared<AMDGCNTargetFusionInfo>(Mod);
    return;
  }
#endif // FUSION_JIT_SUPPORT_AMDGCN
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
                                         BarrierFlags BarrierFlags) const {
  Impl->createBarrierCall(Builder, BarrierFlags);
}

unsigned TargetFusionInfo::getPrivateAddressSpace() const {
  return Impl->getPrivateAddressSpace();
}

unsigned TargetFusionInfo::getLocalAddressSpace() const {
  return Impl->getLocalAddressSpace();
}

void TargetFusionInfo::updateAddressSpaceMetadata(Function *KernelFunc,
                                                  ArrayRef<bool> ArgIsPromoted,
                                                  unsigned AddressSpace) const {
  Impl->updateAddressSpaceMetadata(KernelFunc, ArgIsPromoted, AddressSpace);
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

Error TargetFusionInfo::scanForBuiltinsToRemap(
    Function *F, jit_compiler::Remapper &R,
    const jit_compiler::NDRange &SrcNDRange,
    const jit_compiler::NDRange &FusedNDRange) const {
  return Impl->scanForBuiltinsToRemap(F, R, SrcNDRange, FusedNDRange);
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
    const Remapper &R, BuiltinKind K, Function *F, Module *M,
    const NDRange &SrcNDRange, const NDRange &FusedNDRange) const {
  return Impl->createRemapperFunction(R, K, F->getName(), M, SrcNDRange,
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
