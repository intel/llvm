//==------------------------ SYCLKernelFusion.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLKernelFusion.h"

#include "Kernel.h"
#include "NDRangesHelper.h"
#include "debug/PassDebug.h"
#include "internalization/Internalization.h"
#include "kernel-fusion/Builtins.h"
#include "kernel-info/SYCLKernelInfo.h"
#include "syclcp/SYCLCP.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <sstream>
#include <tuple>

#define DEBUG_TYPE "sycl-fusion"

using namespace llvm;

constexpr StringLiteral SYCLKernelFusion::NDRangeMDKey;
constexpr StringLiteral SYCLKernelFusion::NDRangesMDKey;

struct InputKernel {
  StringRef Name;
  jit_compiler::NDRange ND;
  InputKernel(StringRef Name, const jit_compiler::NDRange &ND)
      : Name{Name}, ND{ND} {}
};

struct InputKernelFunction {
  Function *F;
  const jit_compiler::NDRange &ND;
  InputKernelFunction(Function *F, const jit_compiler::NDRange &ND)
      : F{F}, ND{ND} {}
};

template <typename T> struct GetIntFromMD {
  T operator()(const Metadata *MD) const {
    return cast<ConstantAsMetadata>(MD)
        ->getValue()
        ->getUniqueInteger()
        .getZExtValue();
  }

  unsigned N;
};

static jit_compiler::Indices getIdxFromMD(const Metadata *MD) {
  constexpr unsigned IndicesBitWidth{64};

  jit_compiler::Indices Res;
  const auto *NMD = cast<MDNode>(MD);
  const GetIntFromMD<std::size_t> Trans{IndicesBitWidth};
  std::transform(NMD->op_begin(), NMD->op_end(), Res.begin(),
                 [&](const MDOperand &O) { return Trans(O.get()); });
  return Res;
}

static jit_compiler::NDRange getNDFromMD(const Metadata *MD) {
  const auto *NMD = cast<MDNode>(MD);
  return {GetIntFromMD<int>{32}(NMD->getOperand(0)),
          getIdxFromMD(NMD->getOperand(1)), getIdxFromMD(NMD->getOperand(2)),
          getIdxFromMD(NMD->getOperand(3))};
}

static unsigned getUnsignedFromMD(Metadata *MD);

static bool hasHeterogeneousNDRangesList(ArrayRef<InputKernelFunction> Fs) {
  SmallVector<jit_compiler::NDRange> NDRanges;
  NDRanges.reserve(Fs.size());
  std::transform(Fs.begin(), Fs.end(), std::back_inserter(NDRanges),
                 [](auto &F) { return F.ND; });
  return jit_compiler::isHeterogeneousList(NDRanges);
}

static std::pair<unsigned, unsigned> getKeyFromMD(const MDNode *MD) {
  Metadata *Op0 = MD->getOperand(0).get();
  Metadata *LhsMD;
  Metadata *RhsMD;
  if (isa<ConstantAsMetadata>(Op0)) {
    LhsMD = Op0;
    RhsMD = MD->getOperand(1);
  } else {
    const auto *ParamMD = cast<MDNode>(Op0);
    LhsMD = ParamMD->getOperand(0);
    RhsMD = ParamMD->getOperand(1);
  }
  return std::make_pair<unsigned, unsigned>(getUnsignedFromMD(LhsMD),
                                            getUnsignedFromMD(RhsMD));
}

static void copyArgsMD(
    LLVMContext &LLVMCtx, StringRef KeyID, const Function &Stub,
    Function &Fused,
    const DenseMap<std::pair<unsigned, unsigned>, unsigned> &ParamMapping,
    StringRef DefaultVal = "") {
  const MDNode *MD = Stub.getMetadata(KeyID);
  if (!MD) {
    return;
  }
  SmallVector<Metadata *> NewMD(Fused.arg_size(),
                                MDString::get(LLVMCtx, DefaultVal));
  {
    SmallDenseSet<unsigned> IndexSet;
    for (const auto &Op : MD->operands()) {
      const auto *Node = cast<MDNode>(Op.get());
      const auto Key = getKeyFromMD(Node);
      const auto Iter = ParamMapping.find(Key);
      if (Iter != ParamMapping.end()) {
        const auto Index = Iter->second;
        if (IndexSet.insert(Index).second) {
          NewMD[Index] = Node->getOperand(1);
        }
      }
    }
  }

  Fused.setMetadata(KeyID, MDNode::get(LLVMCtx, NewMD));
}

PreservedAnalyses SYCLKernelFusion::run(Module &M, ModuleAnalysisManager &AM) {
  // Retrieve the SYCLModuleInfo from the corresponding analysis pass
  jit_compiler::SYCLModuleInfo *ModuleInfo =
      AM.getResult<SYCLModuleInfoAnalysis>(M).ModuleInfo;
  assert(ModuleInfo && "No module information available");

  TargetFusionInfo TFI{&M};

  // Iterate over the functions in the module and locate all
  // stub functions identified by metadata.
  SmallPtrSet<Function *, 8> ToCleanUp;
  Error DeferredErrs = Error::success();
  for (Function &F : M) {
    if (F.isDeclaration() // The stub should be a declaration and not defined.
        && F.hasMetadata(
               MetadataKind)) { // Use metadata to identify the stub function.
      // Insert a new function, fusing the kernels listed in the metadata
      // attached to this stub function.
      // The newly created function will carry the name also specified
      // in the metadata.
      if (auto Err = fuseKernel(M, F, ModuleInfo, TFI, ToCleanUp)) {
        DeferredErrs = joinErrors(std::move(DeferredErrs), std::move(Err));
      }
      // Rembember the stub for deletion, as it is not required anymore after
      // inserting the actual fused function.
      ToCleanUp.insert(&F);
    }
  }
  // Notify the target-specific logic that some functions will be erased
  // shortly.
  SmallVector<Function *> NotifyDelete{ToCleanUp.begin(), ToCleanUp.end()};
  TFI.notifyFunctionsDelete(NotifyDelete);
  // Delete all the stub functions
  for (Function *SF : ToCleanUp) {
    SF->eraseFromParent();
  }

  handleAllErrors(std::move(DeferredErrs), [](const StringError &EL) {
    FUSION_DEBUG(dbgs() << EL.message() << "\n");
  });

  // Inserting a new function and deleting the stub function is a major
  // modification to the module and we did not update any analyses,
  // so return none here.
  return ToCleanUp.empty() ? PreservedAnalyses::all()
                           : PreservedAnalyses::none();
}

struct FusionInsertPoints {
  /// The current insert block
  BasicBlock *Entry;
  /// Block for the barrier
  BasicBlock *CallInsertion;
  /// Next insert block
  BasicBlock *Exit;
};

static bool needsGuard(const jit_compiler::NDRange &SrcNDRange,
                       const jit_compiler::NDRange &FusedNDRange) {
  return jit_compiler::NDRange::linearize(SrcNDRange.getGlobalSize()) !=
         jit_compiler::NDRange::linearize(FusedNDRange.getGlobalSize());
}

static FusionInsertPoints addGuard(IRBuilderBase &Builder,
                                   const TargetFusionInfo &TargetInfo,
                                   const jit_compiler::NDRange &SrcNDRange,
                                   const jit_compiler::NDRange &FusedNDRange,
                                   bool IsLast) {
  // Guard:

  // entry:
  //   %g = call GetGlobalLinearIDName
  //   %gicond = cmp ULT, %g, NumWorkItems
  //   br %gicond, call, exit
  //  call:
  //   call Kernel
  //  exit:
  //   call BarrierName
  //    ...
  auto *Entry = Builder.GetInsertBlock();
  if (!needsGuard(SrcNDRange, FusedNDRange)) {
    return {Entry, Entry, Entry};
  }
  // Needs GI guard
  auto &C = Builder.getContext();
  auto *F = Builder.GetInsertBlock()->getParent();

  auto *Exit = BasicBlock::Create(C, "", F);
  auto *CallInsertion = BasicBlock::Create(C, "", F, Exit); // If

  auto *GlobalLinearID =
      jit_compiler::getGlobalLinearID(Builder, TargetInfo, FusedNDRange);

  const auto GI = jit_compiler::NDRange::linearize(SrcNDRange.getGlobalSize());
  auto *Cond = Builder.CreateICmpULT(
      GlobalLinearID,
      Builder.getIntN(TargetInfo.getIndexSpaceBuiltinBitwidth(), GI));

  Builder.CreateCondBr(Cond, CallInsertion, Exit);
  return {Entry, CallInsertion, Exit};
}

static Expected<CallInst *> createFusionCall(
    IRBuilderBase &Builder, Function *F, ArrayRef<Value *> CallArgs,
    const jit_compiler::NDRange &SrcNDRange,
    const jit_compiler::NDRange &FusedNDRange, bool IsLast,
    jit_compiler::BarrierFlags BarriersFlags, jit_compiler::Remapper &Remapper,
    bool ShouldRemap, TargetFusionInfo &TargetInfo) {
  const auto IPs =
      addGuard(Builder, TargetInfo, SrcNDRange, FusedNDRange, IsLast);

  if (ShouldRemap) {
    auto FOrErr = Remapper.remapBuiltins(F, SrcNDRange, FusedNDRange);
    if (auto Err = FOrErr.takeError()) {
      return std::move(Err);
    }
    F = *FOrErr;
  }

  // Insert call
  Builder.SetInsertPoint(IPs.CallInsertion);
  auto *Res = Builder.CreateCall(F, CallArgs);
  Res->setCallingConv(F->getCallingConv());
  Res->setAttributes(F->getAttributes());
  {
    // If we have introduced a guard, branch to barrier.
    auto *BrTarget = IPs.Exit;
    if (IPs.CallInsertion != BrTarget) {
      Builder.CreateBr(BrTarget);
    }
  }

  Builder.SetInsertPoint(IPs.Exit);

  // Insert barrier if needed
  if (!IsLast && !jit_compiler::isNoBarrierFlag(BarriersFlags)) {
    TargetInfo.createBarrierCall(Builder, BarriersFlags);
  }

  // Set insert point for future insertions
  Builder.SetInsertPoint(IPs.Exit);

  return Res;
}

Error SYCLKernelFusion::fuseKernel(
    Module &M, Function &StubFunction, jit_compiler::SYCLModuleInfo *ModInfo,
    TargetFusionInfo &TargetInfo,
    SmallPtrSetImpl<Function *> &ToCleanUp) const {
  // Retrieve the metadata from the stub function.
  // The first operand of the tuple is the name that the newly created,
  // fused function should carry.
  // The second operand is the list (MDTuple) of function names that should be
  // fused into this fused kernel.
  // Fuction names may appear multiple times, resulting in
  // the corresponding function to be included multiple times. The functions
  // are included in the same order as given by the list.
  // If not ND-ranges information is provided, all kernels are assumed
  // to have the same ND-range.
  MDNode *MD = StubFunction.getMetadata(MetadataKind);
  auto *NDRangesMD = StubFunction.getMetadata(NDRangesMDKey);
  auto *NDRangeMD = StubFunction.getMetadata(NDRangeMDKey);
  assert(MD && MD->getNumOperands() == 2 &&
         "Metadata tuple should have two operands");

  // First metadata operand should be the fused kernel name.
  auto *KernelName = cast<MDString>(MD->getOperand(0).get());

  // Second metadata operand should be the list of kernels to fuse.
  auto *KernelList = cast<MDNode>(MD->getOperand(1).get());
  SmallVector<InputKernel> FusedKernels;
  {
    const auto NumKernels = KernelList->getNumOperands();
    assert(NDRangeMD && NumKernels == NDRangesMD->getNumOperands() &&
           "All fused kernels should have ND-ranges information available");
    for (unsigned I = 0; I < NumKernels; ++I) {
      auto *FK = cast<MDString>(KernelList->getOperand(I).get());
      FusedKernels.emplace_back(FK->getString(),
                                getNDFromMD(NDRangesMD->getOperand(I).get()));
    }
  }
  // Function name for the fused kernel.
  StringRef FusedKernelName = KernelName->getString();
  // ND-range for the fused kernel.
  const auto NDRange = getNDFromMD(NDRangeMD);

  // The producer of the input to this pass may be able to determine that
  // the same value is used for multiple input functions in the fused kernel,
  // e.g. when using the output of one kernel as the input to another kernel.
  // This information is also given as metadata, more specifically a list of
  // tuples. Each tuple contains two pairs identifying the two identical
  // parameters, e.g. ((0,1),(2,3)) means that the second argument of the
  // first kernel is identical to the fourth argument to the third kernel.
  // Parse this information from metadata if present.
  SmallVector<ParameterIdentity> ParamIdentities;
  if (StubFunction.hasMetadata(ParameterMDKind)) {
    MDNode *ParamMD = StubFunction.getMetadata(ParameterMDKind);
    for (const auto &Op : ParamMD->operands()) {
      auto *Tuple = cast<MDNode>(Op.get());
      assert(Tuple->getNumOperands() == 2 &&
             "Parameter identities should be given as tuples");
      SYCLKernelFusion::Parameter LHS = getParamFromMD(Tuple->getOperand(0));
      SYCLKernelFusion::Parameter RHS = getParamFromMD(Tuple->getOperand(1));
      ParamIdentities.emplace_back(LHS, RHS);
    }
  }
  // Canonicalize the list of parameter equalities/identities.
  // Removes duplicates, resolves transitive identities (A == B == C)
  // and sorts the list.
  canonicalizeParameters(ParamIdentities);

  LLVMContext &LLVMCtx = M.getContext();
  // Locate all the functions that should be fused into this kernel in the
  // module. The module MUST contain definitions for all functions that should
  // be fused, otherwise this is an error.
  SmallVector<InputKernelFunction> InputFunctions;
  SmallVector<Type *> FusedArguments;
  SmallVector<std::string> FusedArgNames;
  SmallVector<AttributeSet> FusedParamAttributes;
  // We must keep track of some metadata attached to each parameter.
  // Collect it, so it can be attached to the fused function later on.
  MetadataCollection MDCollection{TargetInfo.getKernelMetadataKeys()};

  // Collect information for functor & argument layout and attributes from the
  // input kernels below.
  MutableParamKindList FusedParamKinds;
  MutableArgUsageMask FusedArgUsageMask;
  MutableAttributeList FusedAttributes;

  // Mapping from parameter in an input function (index in the list of input
  // functions and index in the original function) to the argument index in the
  // fused function.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> ParamMapping;
  // The list of identical parameters is sorted, so the relevant entry can
  // always only be the current front.
  SYCLKernelFusion::ParameterIdentity *ParamFront = ParamIdentities.begin();
  unsigned FuncIndex = 0;
  unsigned ArgIndex = 0;
  for (const auto &Fused : FusedKernels) {
    const auto FN = Fused.Name;
    Function *FF = M.getFunction(FN);
    assert(
        FF && !FF->isDeclaration() &&
        "Input function definition for fusion must be present in the module");
    InputFunctions.emplace_back(FF, Fused.ND);
    // Collect argument types from the function to add them to the new
    // function's signature.
    // This also populates the parameter mapping from parameter of one of the
    // input functions to parameter of the fused function.
    AttributeList InputAttrList = FF->getAttributes();
    unsigned ParamIndex = 0;
    SmallVector<bool, 8> UsedArgsMask;
    for (const auto &Arg : FF->args()) {
      int IdenticalIdx = -1;
      if (!ParamIdentities.empty() && FuncIndex == ParamFront->LHS.KernelIdx &&
          ParamIndex == ParamFront->LHS.ParamIdx) {
        // Because ParamIdentity is constructed such that LHS > RHS, the other
        // parameter must already have been processed.
        assert(ParamMapping.count(
            {ParamFront->RHS.KernelIdx, ParamFront->RHS.ParamIdx}));
        unsigned Idx =
            ParamMapping[{ParamFront->RHS.KernelIdx, ParamFront->RHS.ParamIdx}];
        // The SYCL runtime is unaware of the actual type of the parameter and
        // simply compares size and raw bytes to determine identical parameters.
        // In case the value is identical, but the underlying LLVM type is
        // different, ignore the identical parameter.
        if (FusedArguments[Idx] == Arg.getType()) {
          IdenticalIdx = Idx;
        }
      }
      if (IdenticalIdx >= 0) {
        // There is another parameter with identical value. Use the existing
        // mapping of that other parameter and do not add this argument to the
        // fused function.
        ParamMapping.insert({{FuncIndex, ParamIndex}, IdenticalIdx});
        ++ParamFront;
        UsedArgsMask.push_back(false);
      } else {
        // There is no identical parameter, so add a new argument to the fused
        // function and add that to the parameter mapping.
        ParamMapping.insert({{FuncIndex, ParamIndex}, ArgIndex++});
        FusedArguments.push_back(Arg.getType());
        // Add the parameter attributes to the attributes of the fused function.
        FusedParamAttributes.push_back(InputAttrList.getParamAttrs(ParamIndex));
        // Add the argument name with a prefix specifying the original kernel to
        // the list of argument names.
        std::string ArgName = (FN + Twine{"_"} + Arg.getName()).str();
        FusedArgNames.push_back(ArgName);
        UsedArgsMask.push_back(true);
      }
      ++ParamIndex;
    }
    // Add the metadata corresponding to the used arguments to the different
    // lists.
    MDCollection.collectFromFunction(FF, UsedArgsMask);

    // Update the fused kernel's KernelInfo with information from this input
    // kernel.
    assert(ModInfo->hasKernelFor(FN.str()) &&
           "No jit_compiler::SYCLKernelInfo found");
    jit_compiler::SYCLKernelInfo &InputKernelInfo =
        *ModInfo->getKernelFor(FN.str());
    appendKernelInfo(FusedParamKinds, FusedArgUsageMask, FusedAttributes,
                     InputKernelInfo, UsedArgsMask);
    ++FuncIndex;
  }

  // Add the information about the new kernel to the SYCLModuleInfo.
  if (!ModInfo->hasKernelFor(FusedKernelName.str())) {
    assert(FusedParamKinds.size() == FusedArgUsageMask.size());
    jit_compiler::SYCLKernelInfo KI{FusedKernelName.str(),
                                    FusedParamKinds.size()};
    KI.Attributes = KernelAttributeList{FusedAttributes.size()};
    llvm::copy(FusedParamKinds, KI.Args.Kinds.begin());
    llvm::copy(FusedArgUsageMask, KI.Args.UsageMask.begin());
    llvm::copy(FusedAttributes, KI.Attributes.begin());
    ModInfo->addKernel(KI);
  }
  jit_compiler::SYCLKernelInfo &FusedKernelInfo =
      *ModInfo->getKernelFor(FusedKernelName.str());

  // Check that no function with the desired name is already present in the
  // module. LLVM would still be able to insert the function (adding a suffix to
  // the name), but we won't be able to correctly call it by its name at
  // runtime.
  assert(!M.getFunction(KernelName->getString()) &&
         "Function name must not be present in module");

  // Create a new function with the name specified in metadata and
  // all arguments that are required for the functions fused into this kernel.
  // Returning void is always correct, as SYCL kernels mustn't return anything.
  FunctionType *FT = FunctionType::get(Type::getVoidTy(LLVMCtx), FusedArguments,
                                       /* isVarArg*/ false);
  // We create the function with default attributes, i.e. the attributes
  // recorded in the module flags. Parameter attributes will be copied over from
  // their original definition (see below). For the remaining attributes, we
  // rely on deduction by a pass such as 'attributor' or 'function-attr'.
  Function *FusedFunction = Function::createWithDefaultAttr(
      FT, GlobalValue::LinkageTypes::ExternalLinkage,
      M.getDataLayout().getProgramAddressSpace(), KernelName->getString(), &M);
  {
    auto DefaultAttr = FusedFunction->getAttributes();
    // Add uniform function attributes, i.e., attributes with identical value on
    // each input function, to the fused function.
    auto *FirstFunction = InputFunctions.front().F;
    for (const auto &UniformKey : TargetInfo.getUniformKernelAttributes()) {
      if (FirstFunction->hasFnAttribute(UniformKey)) {
        DefaultAttr = DefaultAttr.addFnAttribute(
            LLVMCtx, FirstFunction->getFnAttribute(UniformKey));
      }
    }
    // Add the collected parameter attributes to the fused function.
    // Copying the parameter attributes from their original definition in the
    // input kernels should be safe and they most likely can't be deducted later
    // on, as no caller is present in the module.
    auto FusedFnAttrs =
        AttributeList::get(LLVMCtx, DefaultAttr.getFnAttrs(),
                           DefaultAttr.getRetAttrs(), FusedParamAttributes);
    FusedFunction->setAttributes(FusedFnAttrs);
  }

  {
    // As this metainformation is per-call, not per-function, we must handle it
    // differently.
    constexpr StringLiteral DefaultInternalizationVal{"none"};

    // Retain promote.* metadata
    copyArgsMD(LLVMCtx, SYCLInternalizer::Key, StubFunction, *FusedFunction,
               ParamMapping, DefaultInternalizationVal);
    copyArgsMD(LLVMCtx, SYCLInternalizer::LocalSizeKey, StubFunction,
               *FusedFunction, ParamMapping);
    copyArgsMD(LLVMCtx, SYCLInternalizer::ElemSizeKey, StubFunction,
               *FusedFunction, ParamMapping);
    // and JIT constants
    copyArgsMD(LLVMCtx, SYCLCP::Key, StubFunction, *FusedFunction,
               ParamMapping);
  }

  // Attach names to the arguments. The name includes a prefix for the kernel
  // from which this argument came. Naming the parameters themselves is not
  // necessary for functionality, it just improves readibility for debugging
  // purposes.
  for (const auto &AI : llvm::enumerate(FusedFunction->args())) {
    auto &ArgName = FusedArgNames[AI.index()];
    AI.value().setName(ArgName);
  }
  // Attach the fused metadata collected from the different input
  // kernels to the fused function.
  MDCollection.attachToFunction(FusedFunction);
  // Add metadata for reqd_work_group_size and work_group_size_hint
  attachKernelAttributeMD(LLVMCtx, FusedFunction, FusedKernelInfo);

  // Mark the fused function as a kernel by calling TargetFusionInfo, because
  // this is target-specific.
  TargetInfo.addKernelFunction(FusedFunction);

  // Fusion is implemented as a two step process: In the first step, we
  // simply create calls to the functions that should be fused into this
  // kernel and pass the correct arguments.
  // In the second step, we inline the previously created calls to have
  // the fused functions correctly integrated into the control-flow of
  // the fused function.

  // Create an entry block for the function
  auto *EntryBlock = BasicBlock::Create(LLVMCtx, "entry", FusedFunction);
  IRBuilder<> Builder(LLVMCtx);
  Builder.SetInsertPoint(EntryBlock);
  // Remember calls to inline them later on.
  SmallVector<CallInst *> Calls;
  // Iterate over the functions that should be fused and
  // create a call to each of them (or multiple calls,
  // in case a function should be fused into the kernel
  // multiple times).
  FuncIndex = 0;
  {
    const auto BarriersEnd = InputFunctions.size() - 1;
    const auto IsHeterogeneousNDRangesList =
        hasHeterogeneousNDRangesList(InputFunctions);
    jit_compiler::Remapper Remapper(TargetInfo);

    Error DeferredErrs = Error::success();
    for (auto &KF : InputFunctions) {
      auto *IF = KF.F;
      SmallVector<Value *> CallArgs;
      for (size_t I = 0; I < IF->arg_size(); ++I) {
        // Lookup actual parameter index in the mapping.
        assert(ParamMapping.count({FuncIndex, I}));
        unsigned ParamIdx = ParamMapping[{FuncIndex, I}];
        CallArgs.push_back(FusedFunction->getArg(ParamIdx));
      }
      auto CallOrErr = createFusionCall(
          Builder, IF, CallArgs, KF.ND, NDRange, FuncIndex == BarriersEnd,
          BarriersFlags, Remapper, IsHeterogeneousNDRangesList, TargetInfo);
      // Add to the set of original kernel functions that can be deleted after
      // fusion is complete.
      ToCleanUp.insert(IF);
      ++FuncIndex;

      if (auto Err = CallOrErr.takeError()) {
        DeferredErrs = joinErrors(std::move(DeferredErrs), std::move(Err));
        continue;
      }
      auto *Call = *CallOrErr;
      Calls.push_back(Call);
      ToCleanUp.insert(Call->getCalledFunction());
    }
    if (DeferredErrs) {
      // If we found an error, clean and exit.
      FusedFunction->eraseFromParent();
      return DeferredErrs;
    }
  }
  // Create a void return at the end of the newly created kernel.
  Builder.CreateRetVoid();

  // Iterate over the previously created calls and inline each of them.
  for (CallInst *InlineCall : Calls) {
    InlineFunctionInfo IFI;
    // This inlines with depth 1, i.e., it will only inline the
    // function called here, but none of the functions called
    // from the inlined function.
    InlineResult InlineRes = InlineFunction(*InlineCall, IFI);
    if (!InlineRes.isSuccess()) {
      // In case inlining fails, we only report it, but still carry on.
      // InlineFunction(...) will leave the program in a well-defined state
      // in case it fails, and calling the function is still semantically
      // correct, although it might hinder some optimizations across the borders
      // of the fused functions. We need to prevent deletion of the called
      // function, though.
      auto *Callee = InlineCall->getCalledFunction();
      FUSION_DEBUG(llvm::dbgs() << "WARNING: Inlining of " << Callee->getName()
                                << " failed due to: "
                                << InlineRes.getFailureReason() << '\n');
      ToCleanUp.erase(Callee);
    }
  }

  // Perform target-specific post-processing of the new fused kernel.
  TargetInfo.postProcessKernel(FusedFunction);
  return Error::success();
}

void SYCLKernelFusion::canonicalizeParameters(
    SmallVectorImpl<ParameterIdentity> &Params) const {
  // Canonicalize the list of parameter identities/equalities.
  // The input is a list of parameter pairs which werde detected to be
  // identical. Each pair is constructed such that the RHS belongs to a kernel
  // occuring before the kernel for the LHS in the list of kernels to fuse. This
  // means, that we want to use the LHS parameter instead of the RHS parameter.

  // In the first step we sort the list of pairs by their LHS.
  std::sort(Params.begin(), Params.end());

  // Iterate through the list and handle two cases:
  // 1. Duplicates: ((K2,P2),(K1, P2)) and ((K2, P2),(K1, P5)) both in the list
  // In this case, remove ((K2, P2),(K1, P5)) from the list and
  // add ((K1, P5),(K1, P2)) as per transitivity.
  // 2. Transitivity: ((K2, P2),(K1,P3)) and ((K3,P4),(K2,P2)) both in the list
  // In this case, remove ((K3,P4),(K2,P2)) and add ((K3,P4),(K1,P3))
  // as per transitivity.
  std::map<Parameter, Parameter> Identities;
  SmallVector<ParameterIdentity> NewEntries;
  for (auto *I = Params.begin(); I != Params.end();) {
    if (I->LHS == I->RHS) {
      // LHS and RHS are identical - this does not provide
      // any useful information at all, discard it.
      I = Params.erase(I);
    }
    if (Identities.count(I->LHS)) {
      // Duplicate
      auto ExistingIdentity = Identities.at(I->LHS);
      Identities.emplace(I->RHS, ExistingIdentity);
      NewEntries.emplace_back(I->RHS, ExistingIdentity);
      I = Params.erase(I);
    } else if (Identities.count(I->RHS)) {
      // Transitivity
      auto ExistingIdentity = Identities.at(I->RHS);
      Identities.emplace(I->RHS, ExistingIdentity);
      NewEntries.emplace_back(I->RHS, ExistingIdentity);
      I = Params.erase(I);
    } else {
      Identities.emplace(I->LHS, I->RHS);
      ++I;
    }
  }
  // Append the new entries to the list.
  Params.append(NewEntries.begin(), NewEntries.end());

  // Sort the list again - this way, we can simply consume it
  // step by step instead of having to perform a map-like lookup.
  std::sort(Params.begin(), Params.end());
}

SYCLKernelFusion::Parameter
SYCLKernelFusion::getParamFromMD(Metadata *MD) const {
  // Parse a parameter (kernel index + parameter index) from metadata.
  assert(MD && "Empty Metadata");
  auto *Node = cast<MDNode>(MD);
  unsigned KernelIdx = getUnsignedFromMD(Node->getOperand(0));
  unsigned ParamIdx = getUnsignedFromMD(Node->getOperand(1));
  return Parameter{KernelIdx, ParamIdx};
}

static unsigned getUnsignedFromMD(Metadata *MD) {
  // Peel a constant integer from metadata. It is stored
  // as ConstantInt inside a ConstantAsMetadata, a
  // subtype of metadata.
  assert(MD);
  auto *ConstantMD = cast<ConstantAsMetadata>(MD);
  Constant *ConstantVal = ConstantMD->getValue();
  auto *ConstInt = cast<ConstantInt>(ConstantVal);
  return ConstInt->getZExtValue();
}

void SYCLKernelFusion::attachFusedMetadata(
    Function *FusedFunction, const StringRef &Kind,
    const ArrayRef<Metadata *> FusedMetadata) const {
  // Attach a list of fused metadata for a kind to the fused function.
  auto *MDEntries = MDNode::get(FusedFunction->getContext(), FusedMetadata);
  FusedFunction->setMetadata(Kind, MDEntries);
}

void SYCLKernelFusion::attachKernelAttributeMD(
    LLVMContext &LLVMCtx, Function *FusedFunction,
    jit_compiler::SYCLKernelInfo &FusedKernelInfo) const {
  // Attach kernel attribute information as metadata to a kernel function.
  for (jit_compiler::SYCLKernelAttribute &KernelAttr :
       FusedKernelInfo.Attributes) {
    if (KernelAttr.Kind == KernelAttrKind::ReqdWorkGroupSize ||
        KernelAttr.Kind == KernelAttrKind::WorkGroupSizeHint) {
      // 'reqd_work_group_size' and 'work_group_size_hint' get attached as
      // metadata with their three values as constant integer metadata.
      SmallVector<Metadata *, 3> MDValues;
      for (auto Val : KernelAttr.Values) {
        MDValues.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(LLVMCtx), Val)));
      }
      attachFusedMetadata(FusedFunction, KernelAttr.getName(), MDValues);
    }
    // The two kernel attributes above are currently the only attributes
    // attached as metadata, so we don't do anything for other attributes.
  }
}

void SYCLKernelFusion::updateArgUsageMask(
    MutableArgUsageMask &NewMask,
    jit_compiler::SYCLArgumentDescriptor &InputDef,
    const ArrayRef<bool> ParamUseMask) const {
  // Create a new argument usage mask from the input information and the mask
  // resulting from identical parameters. The input kernel may have had unused
  // parameters before and during fusion, more parameters can become unused due
  // to identical parameters. They are still present in the KernelInfo, but
  // not on the kernel function.
  const auto *MaskIt = ParamUseMask.begin();
  for (bool ParamInfo : InputDef.UsageMask) {
    if (!ParamInfo) {
      // The parameter was already unused in the input kernel,
      // so it will be unused here, too.
      NewMask.push_back(false);
    } else {
      // The parameter was used on the input kernel,
      // but may have become unused in the fused kernel.
      // Check using the mask.
      NewMask.push_back(*(MaskIt++));
    }
  }
}

void SYCLKernelFusion::appendKernelInfo(
    MutableParamKindList &FusedParamKinds,
    MutableArgUsageMask &FusedArgUsageMask,
    MutableAttributeList &FusedAttributes,
    jit_compiler::SYCLKernelInfo &InputInfo,
    const ArrayRef<bool> ParamUseMask) const {
  // Add information from the input kernel to the SYCLKernelInfo of the fused
  // kernel.

  // Add information about the input kernel's arguments to the KernelInfo for
  // the fused function.
  FusedParamKinds.append(InputInfo.Args.Kinds.begin(),
                         InputInfo.Args.Kinds.end());

  // Create a argument usage mask from input information and the mask resulting
  // from potential identical parameters.
  MutableArgUsageMask NewMask;
  updateArgUsageMask(NewMask, InputInfo.Args, ParamUseMask);
  FusedArgUsageMask.append(NewMask.begin(), NewMask.end());

  // Merge the existing kernel attributes for the fused kernel (potentially
  // still empty) with the kernel attributes of the input kernel.
  mergeKernelAttributes(FusedAttributes, InputInfo.Attributes);
}

void SYCLKernelFusion::mergeKernelAttributes(
    MutableAttributeList &Attributes, const KernelAttributeList &Other) const {
  // For the current set of valid kernel attributes, it is sufficient to only
  // iterate over the list of new attributes coming in. In cases where the
  // existing list contains an attribute that is not present in the new list, we
  // want to keep it anyways.
  for (const jit_compiler::SYCLKernelAttribute &OtherAttr : Other) {
    SYCLKernelFusion::KernelAttr *Attr =
        getAttribute(Attributes, OtherAttr.Kind);
    SYCLKernelFusion::AttrMergeResult MergeResult =
        mergeAttribute(Attr, OtherAttr);
    switch (MergeResult) {
    case AttrMergeResult::KeepAttr: /* Nothing to do, just keep the result */
      break;
    case AttrMergeResult::UpdateAttr: /* Attribute should have been updated in
                                         place */
      break;
    case AttrMergeResult::AddAttr:
      addAttribute(Attributes, OtherAttr);
      break;
    case AttrMergeResult::RemoveAttr:
      removeAttribute(Attributes, OtherAttr.Kind);
      break;
    case AttrMergeResult::Error:
      llvm_unreachable("Failed to merge attribute");
      break;
    }
  }
}

SYCLKernelFusion::AttrMergeResult
SYCLKernelFusion::mergeAttribute(KernelAttr *Attr,
                                 const KernelAttr &Other) const {
  switch (Other.Kind) {
  case KernelAttrKind::ReqdWorkGroupSize:
    return mergeReqdWorkgroupSize(Attr, Other);
  case KernelAttrKind::WorkGroupSizeHint:
    return mergeWorkgroupSizeHint(Attr, Other);
  default:
    // Unknown attribute name, return an error.
    return SYCLKernelFusion::AttrMergeResult::Error;
  }
}

SYCLKernelFusion::AttrMergeResult
SYCLKernelFusion::mergeReqdWorkgroupSize(KernelAttr *Attr,
                                         const KernelAttr &Other) const {
  if (!Attr) {
    // The existing list does not contain a hint for the workgroup size, add the
    // new one
    return SYCLKernelFusion::AttrMergeResult::AddAttr;
  }
  if (Attr->Values != Other.Values) {
    // Two different required work-group sizes, causes an error.
    return SYCLKernelFusion::AttrMergeResult::Error;
  }
  // The required workgroup sizes are identical, keep it.
  return SYCLKernelFusion::AttrMergeResult::KeepAttr;
}

SYCLKernelFusion::AttrMergeResult
SYCLKernelFusion::mergeWorkgroupSizeHint(KernelAttr *Attr,
                                         const KernelAttr &Other) const {
  if (!Attr) {
    // The existing list does not contain a hint for the workgroup size, add
    // the new one
    return SYCLKernelFusion::AttrMergeResult::AddAttr;
  }
  if (Attr->Values != Other.Values) {
    // Two different hints, remove the hint altogether.
    return SYCLKernelFusion::AttrMergeResult::RemoveAttr;
  }
  // The given hint is identical, keep it.
  return SYCLKernelFusion::AttrMergeResult::KeepAttr;
}

SYCLKernelFusion::KernelAttr *
SYCLKernelFusion::getAttribute(MutableAttributeList &Attributes,
                               KernelAttrKind AttrKind) const {
  auto *It = findAttribute(Attributes, AttrKind);
  if (It != Attributes.end()) {
    return &*It;
  }
  return nullptr;
}

void SYCLKernelFusion::addAttribute(MutableAttributeList &Attributes,
                                    const KernelAttr &Attr) const {
  Attributes.push_back(Attr);
}

void SYCLKernelFusion::removeAttribute(MutableAttributeList &Attributes,
                                       KernelAttrKind AttrKind) const {
  auto *It = findAttribute(Attributes, AttrKind);
  if (It != Attributes.end()) {
    Attributes.erase(It);
  }
}

SYCLKernelFusion::MutableAttributeList::iterator
SYCLKernelFusion::findAttribute(MutableAttributeList &Attributes,
                                KernelAttrKind AttrKind) const {
  return llvm::find_if(Attributes, [=](SYCLKernelFusion::KernelAttr &Attr) {
    return Attr.Kind == AttrKind;
  });
}
