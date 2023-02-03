//==------------------------ SYCLKernelFusion.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLKernelFusion.h"

#include "debug/PassDebug.h"
#include "internalization/Internalization.h"
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

constexpr static StringLiteral KernelArgAddrSpace{"kernel_arg_addr_space"};
constexpr static StringLiteral KernelArgAccessQual{"kernel_arg_access_qual"};
constexpr static StringLiteral KernelArgType{"kernel_arg_type"};
constexpr static StringLiteral KernelArgBaseType{"kernel_arg_base_type"};
constexpr static StringLiteral KernelArgTypeQual{"kernel_arg_type_qual"};

static unsigned getUnsignedFromMD(Metadata *MD);

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

  // Iterate over the functions in the module and locate all
  // stub functions identified by metadata.
  SmallPtrSet<Function *, 8> ToCleanUp;
  for (Function &F : M) {
    if (F.isDeclaration() // The stub should be a declaration and not defined.
        && F.hasMetadata(
               MetadataKind)) { // Use metadata to identify the stub function.
      // Insert a new function, fusing the kernels listed in the metadata
      // attached to this stub function.
      // The newly created function will carry the name also specified
      // in the metadata.
      fuseKernel(M, F, ModuleInfo, ToCleanUp);
      // Rembember the stub for deletion, as it is not required anymore after
      // inserting the actual fused function.
      ToCleanUp.insert(&F);
    }
  }
  // Delete all the stub functions
  for (Function *SF : ToCleanUp) {
    SF->eraseFromParent();
  }
  // Inserting a new function and deleting the stub function is a major
  // modification to the module and we did not update any analyses,
  // so return none here.
  return ToCleanUp.empty() ? PreservedAnalyses::all()
                           : PreservedAnalyses::none();
}

struct FunctionCall {
  FunctionCallee Callee;
  SmallVector<Value *> Args;
};

template <typename T> static void setBarrierMetadata(T *Ptr, LLVMContext &C) {
  const auto FnAttrs =
      AttributeSet::get(C, {Attribute::get(C, Attribute::AttrKind::Convergent),
                            Attribute::get(C, Attribute::AttrKind::NoUnwind)});
  Ptr->setAttributes(AttributeList::get(C, FnAttrs, {}, {}));
  Ptr->setCallingConv(CallingConv::SPIR_FUNC);
}

static FunctionCall createBarrierCall(IRBuilderBase &Builder, Module &M,
                                      int Flags) {
  assert((Flags == 1 || Flags == 2 || Flags == 3) && "Invalid barrier flags");

  constexpr StringLiteral N{"_Z22__spirv_ControlBarrierjjj"};

  Function *F = M.getFunction(N);
  if (!F) {
    constexpr auto Linkage = GlobalValue::LinkageTypes::ExternalLinkage;

    auto *Ty = FunctionType::get(
        Builder.getVoidTy(),
        {Builder.getInt32Ty(), Builder.getInt32Ty(), Builder.getInt32Ty()},
        false /* isVarArg*/);

    F = Function::Create(Ty, Linkage, N, M);

    setBarrierMetadata(F, Builder.getContext());
  }

  // See
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-
  return {F,
          {Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
           Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
           Builder.getInt32(0x10 | (Flags % 2 == 1 ? 0x100 : 0x0) |
                            ((Flags >> 1 == 1 ? 0x200 : 0x0)))}};
}

void SYCLKernelFusion::fuseKernel(
    Module &M, Function &StubFunction, jit_compiler::SYCLModuleInfo *ModInfo,
    SmallPtrSetImpl<Function *> &ToCleanUp) const {
  // Retrieve the metadata from the stub function.
  // The first operand of the tuple is the name that the newly created,
  // fused function should carry.
  // The second operand is the list (MDTuple) of function names that should be
  // fused into this fused kernel.
  // Fuction names may appear multiple times, resulting in
  // the corresponding function to be included multiple times. The functions
  // are included in the same order as given by the list.
  MDNode *MD = StubFunction.getMetadata(MetadataKind);
  assert(MD && MD->getNumOperands() == 2 &&
         "Metadata tuple should have two operands");

  // First metadata operand should be the fused kernel name.
  auto *KernelName = cast<MDString>(MD->getOperand(0).get());

  // Second metadata operand should be the list of kernels to fuse.
  auto *KernelList = cast<MDNode>(MD->getOperand(1).get());
  SmallVector<StringRef> FusedKernels;
  for (const MDOperand &MDOp : KernelList->operands()) {
    // Kernel should be given by its name as MDString
    auto *FK = cast<MDString>(MDOp.get());
    FusedKernels.push_back(FK->getString());
  }
  // Function name for the fused kernel.
  StringRef FusedKernelName = KernelName->getString();

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
  SmallVector<Function *> InputFunctions;
  SmallVector<Type *> FusedArguments;
  SmallVector<std::string> FusedArgNames;
  SmallVector<AttributeSet> FusedParamAttributes;
  // We must keep track of some metadata attached to each parameter.
  // Collect it in lists, so it can be attached to the fused function later on.
  MDList KernelArgAddressSpaces;
  MDList KernelArgAccessQualifiers;
  MDList KernelArgTypes;
  MDList KernelArgBaseTypes;
  MDList KernelArgTypeQualifiers;
  // Add the information about the new kernel to the SYCLModuleInfo.
  // Initialize the jit_compiler::SYCLKernelInfo with the name. The remaining
  // information for functor & argument layout and attributes will be filled in
  // with information from the input kernels below.
  if (!ModInfo->hasKernelFor(FusedKernelName.str())) {
    jit_compiler::SYCLKernelInfo KI{FusedKernelName.str()};
    ModInfo->addKernel(KI);
  }
  jit_compiler::SYCLKernelInfo &FusedKernelInfo =
      *ModInfo->getKernelFor(FusedKernelName.str());
  // Mapping from parameter in an input function (index in the list of input
  // functions and index in the original function) to the argument index in the
  // fused function.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> ParamMapping;
  // The list of identical parameters is sorted, so the relevant entry can
  // always only be the current front.
  SYCLKernelFusion::ParameterIdentity *ParamFront = ParamIdentities.begin();
  unsigned FuncIndex = 0;
  unsigned ArgIndex = 0;
  for (StringRef &FN : FusedKernels) {
    Function *FF = M.getFunction(FN);
    assert(
        FF && !FF->isDeclaration() &&
        "Input function definition for fusion must be present in the module");
    InputFunctions.push_back(FF);
    // Collect argument types from the function to add them to the new
    // function's signature.
    // This also populates the parameter mapping from parameter of one of the
    // input functions to parameter of the fused function.
    AttributeList InputAttrList = FF->getAttributes();
    unsigned ParamIndex = 0;
    SmallVector<bool, 8> UsedArgsMask;
    for (const auto &Arg : FF->args()) {
      if (!ParamIdentities.empty() && FuncIndex == ParamFront->LHS.KernelIdx &&
          ParamIndex == ParamFront->LHS.ParamIdx) {
        // There is another parameter with identical value. Use the existing
        // mapping of that other parameter and do not add this argument to the
        // fused function. Because ParamIdentity is constructed such that LHS >
        // RHS, the other parameter must already have been processed.
        assert(ParamMapping.count(
            {ParamFront->RHS.KernelIdx, ParamFront->RHS.ParamIdx}));
        unsigned Idx =
            ParamMapping[{ParamFront->RHS.KernelIdx, ParamFront->RHS.ParamIdx}];
        ParamMapping.insert({{FuncIndex, ParamIndex}, Idx});
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
    // lists. NOTE: We do not collect the "kernel_arg_name" metadata, because
    // the kernel arguments receive new names in the fused kernel.
    addToFusedMetadata(FF, KernelArgAddrSpace, UsedArgsMask,
                       KernelArgAddressSpaces);
    addToFusedMetadata(FF, KernelArgAccessQual, UsedArgsMask,
                       KernelArgAccessQualifiers);
    addToFusedMetadata(FF, KernelArgType, UsedArgsMask, KernelArgTypes);
    addToFusedMetadata(FF, KernelArgBaseType, UsedArgsMask, KernelArgBaseTypes);
    addToFusedMetadata(FF, KernelArgTypeQual, UsedArgsMask,
                       KernelArgTypeQualifiers);

    // Update the fused kernel's KernelInfo with information from this input
    // kernel.
    assert(ModInfo->hasKernelFor(FN.str()) &&
           "No jit_compiler::SYCLKernelInfo found");
    jit_compiler::SYCLKernelInfo &InputKernelInfo =
        *ModInfo->getKernelFor(FN.str());
    appendKernelInfo(FusedKernelInfo, InputKernelInfo, UsedArgsMask);
    ++FuncIndex;
  }

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
    // Add the collected parameter attributes to the fused function.
    // Copying the parameter attributes from their original definition in the
    // input kernels should be safe and they most likely can't be deducted later
    // on, as no caller is present in the module.
    auto DefaultAttr = FusedFunction->getAttributes();
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
    // and JIT constants
    copyArgsMD(LLVMCtx, SYCLCP::Key, StubFunction, *FusedFunction,
               ParamMapping);
  }

  // Attach names to the arguments. The name includes a prefix for the kernel
  // from which this argument came. The names are also attached as metadata
  // with kind "kernel_arg_name".
  // NOTE: While the kernel_arg_name metadata is required, naming the
  // parameters themselves is not necessary for functionality, it just improves
  // readibility for debugging purposes.
  SmallVector<Metadata *, 16> KernelArgNames;
  for (const auto &AI : llvm::enumerate(FusedFunction->args())) {
    auto &ArgName = FusedArgNames[AI.index()];
    AI.value().setName(ArgName);
    KernelArgNames.push_back(MDString::get(LLVMCtx, ArgName));
  }
  // Attach the fused kernel_arg_* metadata collected from the different input
  // kernels to the fused function.
  attachFusedMetadata(FusedFunction, "kernel_arg_addr_space",
                      KernelArgAddressSpaces);
  attachFusedMetadata(FusedFunction, "kernel_arg_access_qual",
                      KernelArgAccessQualifiers);
  attachFusedMetadata(FusedFunction, "kernel_arg_type", KernelArgTypes);
  attachFusedMetadata(FusedFunction, "kernel_arg_base_type",
                      KernelArgBaseTypes);
  attachFusedMetadata(FusedFunction, "kernel_arg_type_qual",
                      KernelArgTypeQualifiers);
  attachFusedMetadata(FusedFunction, "kernel_arg_name", KernelArgNames);
  // Add metadata for reqd_work_group_size and work_group_size_hint
  attachKernelAttributeMD(LLVMCtx, FusedFunction, FusedKernelInfo);

  // The fused kernel should be a SPIR-V kernel again.
  // NOTE: If this pass is used in a scenario where input and output
  // of the compilation are not SPIR-V, care must be taken of other
  // potential calling conventions here (e.g., nvptx).
  FusedFunction->setCallingConv(CallingConv::SPIR_KERNEL);

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
    const auto BarrierCall =
        BarriersFlags != -1 ? std::optional<FunctionCall>{createBarrierCall(
                                  Builder, M, BarriersFlags)}
                            : std::optional<FunctionCall>{};
    const auto BarriersEnd = InputFunctions.size() - 1;

    for (Function *IF : InputFunctions) {
      SmallVector<Value *> CallArgs;
      for (size_t I = 0; I < IF->arg_size(); ++I) {
        // Lookup actual parameter index in the mapping.
        assert(ParamMapping.count({FuncIndex, I}));
        unsigned ParamIdx = ParamMapping[{FuncIndex, I}];
        CallArgs.push_back(FusedFunction->getArg(ParamIdx));
      }
      Calls.push_back(Builder.CreateCall(IF, CallArgs));
      if (BarriersFlags != -1 && FuncIndex < BarriersEnd) {
        auto *BarrierCallInst =
            Builder.CreateCall(BarrierCall->Callee, BarrierCall->Args);
        setBarrierMetadata(BarrierCallInst, Builder.getContext());
      }
      ++FuncIndex;
      // Add to the set of original kernel functions that can be deleted after
      // fusion is complete.
      ToCleanUp.insert(IF);
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
      // of the fused functions.
      FUSION_DEBUG(llvm::dbgs()
                   << "WARNING: Inlining of "
                   << InlineCall->getCalledFunction()->getName()
                   << " failed due to: " << InlineRes.getFailureReason());
    }
  }

  // Remove all existing calls of the ITT instrumentation functions. Insert new
  // ones in the entry block of the fused kernel and every exit block if the
  // functions are present in the module.
  // We cannot use the existing SPIRITTAnnotations pass, because that pass might
  // insert calls to functions not present in the module (e.g., ITT
  // instrumentations for barriers). As the JITed module is not linked with
  // libdevice anymore, the functions would remain unresolved and cause the
  // driver to fail.
  Function *StartWrapperFunc = M.getFunction(ITTStartWrapper);
  Function *FinishWrapperFunc = M.getFunction(ITTFinishWrapper);
  bool InsertWrappers =
      ((StartWrapperFunc && !StartWrapperFunc->isDeclaration()) &&
       (FinishWrapperFunc && !FinishWrapperFunc->isDeclaration()));
  auto *WrapperFuncTy =
      FunctionType::get(Type::getVoidTy(M.getContext()), /*isVarArg*/ false);
  for (auto &BB : *FusedFunction) {
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
    FusedFunction->getEntryBlock().getFirstInsertionPt();
    auto *WrapperCall = CallInst::Create(
        WrapperFuncTy, StartWrapperFunc, "",
        &*FusedFunction->getEntryBlock().getFirstInsertionPt());
    WrapperCall->setCallingConv(CallingConv::SPIR_FUNC);
  }
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

void SYCLKernelFusion::addToFusedMetadata(
    Function *InputFunction, const StringRef &Kind,
    const ArrayRef<bool> IsArgPresentMask,
    SmallVectorImpl<Metadata *> &FusedMDList) const {
  // Retrieve metadata from one of the input kernels and add it to the list
  // of fused metadata.
  assert(InputFunction->hasMetadata(Kind) &&
         "Required Metadata not present on input kernel");
  if (auto *MD = InputFunction->getMetadata(Kind)) {
    for (auto MaskedOps : llvm::zip(IsArgPresentMask, MD->operands())) {
      if (std::get<0>(MaskedOps)) {
        FusedMDList.emplace_back(std::get<1>(MaskedOps).get());
      }
    }
  }
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
    if (KernelAttr.AttributeName == "reqd_work_group_size" ||
        KernelAttr.AttributeName == "work_group_size_hint") {
      // 'reqd_work_group_size' and 'work_group_size_hint' get attached as
      // metadata with their three values as constant integer metadata.
      SmallVector<Metadata *, 3> MDValues;
      for (std::string &Val : KernelAttr.Values) {
        MDValues.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(LLVMCtx), std::stoi(Val))));
      }
      attachFusedMetadata(FusedFunction, KernelAttr.AttributeName, MDValues);
    }
    // The two kernel attributes above are currently the only attributes
    // attached as metadata, so we don't do anything for other attributes.
  }
}

void SYCLKernelFusion::updateArgUsageMask(
    jit_compiler::ArgUsageMask &NewMask,
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
    jit_compiler::SYCLKernelInfo &FusedInfo,
    jit_compiler::SYCLKernelInfo &InputInfo,
    const ArrayRef<bool> ParamUseMask) const {
  // Add information from the input kernel to the SYCLKernelInfo of the fused
  // kernel.

  // Add information about the input kernel's arguments to the KernelInfo for
  // the fused function.
  FusedInfo.Args.Kinds.insert(FusedInfo.Args.Kinds.end(),
                              InputInfo.Args.Kinds.begin(),
                              InputInfo.Args.Kinds.end());

  // Create a argument usage mask from input information and the mask resulting
  // from potential identical parameters.
  jit_compiler::ArgUsageMask NewMask;
  updateArgUsageMask(NewMask, InputInfo.Args, ParamUseMask);
  FusedInfo.Args.UsageMask.insert(FusedInfo.Args.UsageMask.end(),
                                  NewMask.begin(), NewMask.end());

  // Merge the existing kernel attributes for the fused kernel (potentially
  // still empty) with the kernel attributes of the input kernel.
  mergeKernelAttributes(FusedInfo.Attributes, InputInfo.Attributes);
}

void SYCLKernelFusion::mergeKernelAttributes(
    KernelAttributeList &Attributes, const KernelAttributeList &Other) const {
  // For the current set of valid kernel attributes, it is sufficient to only
  // iterate over the list of new attributes coming in. In cases where the
  // existing list contains an attribute that is not present in the new list, we
  // want to keep it anyways.
  for (const jit_compiler::SYCLKernelAttribute &OtherAttr : Other) {
    SYCLKernelFusion::KernelAttr *Attr =
        getAttribute(Attributes, OtherAttr.AttributeName);
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
      removeAttribute(Attributes, OtherAttr.AttributeName);
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
  if (Other.AttributeName == "reqd_work_group_size") {
    return mergeReqdWorkgroupSize(Attr, Other);
  }
  if (Other.AttributeName == "work_group_size_hint") {
    return mergeWorkgroupSizeHint(Attr, Other);
  }
  // Unknown attribute name, return an error.
  return SYCLKernelFusion::AttrMergeResult::Error;
}

SYCLKernelFusion::AttrMergeResult
SYCLKernelFusion::mergeReqdWorkgroupSize(KernelAttr *Attr,
                                         const KernelAttr &Other) const {
  if (!Attr) {
    // The existing list does not contain a hint for the workgroup size, add the
    // new one
    return SYCLKernelFusion::AttrMergeResult::AddAttr;
  }
  for (size_t I = 0; I < 3; ++I) {
    if (getAttrValueAsInt(*Attr, I) != getAttrValueAsInt(Other, I)) {
      // Two different required work-group sizes, causes an error.
      return SYCLKernelFusion::AttrMergeResult::Error;
    }
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
  for (size_t I = 0; I < 3; ++I) {
    if (getAttrValueAsInt(*Attr, I) != getAttrValueAsInt(Other, I)) {
      // Two different hints, remove the hint altogether.
      return SYCLKernelFusion::AttrMergeResult::RemoveAttr;
    }
  }
  // The given hint is identical, keep it.
  return SYCLKernelFusion::AttrMergeResult::KeepAttr;
}

SYCLKernelFusion::KernelAttr *
SYCLKernelFusion::getAttribute(KernelAttributeList &Attributes,
                               StringRef AttrName) const {
  SYCLKernelFusion::KernelAttrIterator It = findAttribute(Attributes, AttrName);
  if (It != Attributes.end()) {
    return &*It;
  }
  return nullptr;
}

void SYCLKernelFusion::addAttribute(KernelAttributeList &Attributes,
                                    const KernelAttr &Attr) const {
  Attributes.push_back(Attr);
}

void SYCLKernelFusion::removeAttribute(KernelAttributeList &Attributes,
                                       StringRef AttrName) const {
  SYCLKernelFusion::KernelAttrIterator It = findAttribute(Attributes, AttrName);
  if (It != Attributes.end()) {
    Attributes.erase(It);
  }
}

SYCLKernelFusion::KernelAttrIterator
SYCLKernelFusion::findAttribute(KernelAttributeList &Attributes,
                                StringRef AttrName) const {
  return llvm::find_if(Attributes, [=](SYCLKernelFusion::KernelAttr &Attr) {
    return Attr.AttributeName == AttrName.str();
  });
}

unsigned SYCLKernelFusion::getAttrValueAsInt(const KernelAttr &Attr,
                                             size_t Idx) const {
  assert(Idx < Attr.Values.size());
  unsigned Result = 0;
  StringRef(Attr.Values[Idx]).getAsInteger(0, Result);
  return Result;
}
