//===--------- GlobalOffset.cpp - Global Offset Support for CUDA --------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/GlobalOffset.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "globaloffset"

#include "llvm/Support/CommandLine.h"
static cl::opt<bool>
    EnableGlobalOffset("enable-global-offset", cl::Hidden, cl::init(true),
                       cl::desc("Enable SYCL global offset pass"));
namespace llvm {
ModulePass *createGlobalOffsetPass();
void initializeGlobalOffsetPass(PassRegistry &);
} // end namespace llvm

// Legacy PM wrapper.
namespace {
class GlobalOffsetLegacy : public ModulePass {
public:
  static char ID;
  GlobalOffsetLegacy() : ModulePass(ID) {
    initializeGlobalOffsetLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager DummyMAM;
    auto PA = Impl.run(M, DummyMAM);
    return !PA.areAllPreserved();
  }

private:
  GlobalOffsetPass Impl;
};
} // namespace

char GlobalOffsetLegacy::ID = 0;
INITIALIZE_PASS(GlobalOffsetLegacy, "globaloffset",
                "SYCL Add Implicit Global Offset", false, false)

ModulePass *llvm::createGlobalOffsetPassLegacy() {
  return new GlobalOffsetLegacy();
}

// New PM implementation.
PreservedAnalyses GlobalOffsetPass::run(Module &M, ModuleAnalysisManager &) {
  if (!EnableGlobalOffset)
    return PreservedAnalyses::all();

  AT = TargetHelpers::getArchType(M);
  Function *ImplicitOffsetIntrinsic = M.getFunction(Intrinsic::getName(
      AT == ArchType::Cuda
          ? static_cast<unsigned>(Intrinsic::nvvm_implicit_offset)
          : static_cast<unsigned>(Intrinsic::amdgcn_implicit_offset)));

  if (!ImplicitOffsetIntrinsic || ImplicitOffsetIntrinsic->use_empty())
    return PreservedAnalyses::all();

  // For AMD allocas and pointers have to be to CONSTANT_PRIVATE (5), NVVM is
  // happy with ADDRESS_SPACE_GENERIC (0).
  TargetAS = AT == ArchType::Cuda ? 0 : 5;
  /// The value for NVVM's ADDRESS_SPACE_SHARED and AMD's LOCAL_ADDRESS happen
  /// to be 3, use it for the implicit argument pointer type.
  KernelImplicitArgumentType =
      ArrayType::get(Type::getInt32Ty(M.getContext()), 3);
  ImplicitOffsetPtrType =
      Type::getInt32Ty(M.getContext())->getPointerTo(TargetAS);
  assert((ImplicitOffsetIntrinsic->getReturnType() == ImplicitOffsetPtrType) &&
         "Implicit offset intrinsic does not return the expected type");

  SmallVector<KernelPayload, 4> KernelPayloads;
  TargetHelpers::populateKernels(M, KernelPayloads, AT);

  // Validate kernels and populate entry map
  EntryPointMetadata = generateKernelMDNodeMap(M, KernelPayloads);

  // Add implicit parameters to all direct and indirect users of the offset
  addImplicitParameterToCallers(M, ImplicitOffsetIntrinsic, nullptr);

  // Assert that all uses of `ImplicitOffsetIntrinsic` are removed and delete
  // it.
  assert(ImplicitOffsetIntrinsic->use_empty() &&
         "Not all uses of intrinsic removed");
  ImplicitOffsetIntrinsic->eraseFromParent();

  return PreservedAnalyses::none();
}

void GlobalOffsetPass::processKernelEntryPoint(Function *Func) {
  assert(EntryPointMetadata.count(Func) != 0 &&
         "Function must be an entry point");

  auto &M = *Func->getParent();
  LLVMContext &Ctx = M.getContext();
  MDNode *FuncMetadata = EntryPointMetadata[Func];

  // Already processed.
  if (ProcessedFunctions.count(Func) == 1)
    return;

  // Add the new argument to all other kernel entry points, despite not
  // using the global offset.
  auto *KernelMetadata = M.getNamedMetadata(getAnnotationString(AT).c_str());
  assert(KernelMetadata && "IR compiled must have correct annotations");

  auto *NewFunc = addOffsetArgumentToFunction(
                      M, Func, KernelImplicitArgumentType->getPointerTo(),
                      /*KeepOriginal=*/true)
                      .first;
  Argument *NewArgument = std::prev(NewFunc->arg_end());
  // Pass byval to the kernel for NVIDIA, AMD's calling convention disallows
  // byval args, use byref.
  auto Attr =
      AT == ArchType::Cuda
          ? Attribute::getWithByValType(Ctx, KernelImplicitArgumentType)
          : Attribute::getWithByRefType(Ctx, KernelImplicitArgumentType);
  NewArgument->addAttr(Attr);

  // Add the metadata.
  Metadata *NewMetadata[] = {ConstantAsMetadata::get(NewFunc),
                             FuncMetadata->getOperand(1),
                             FuncMetadata->getOperand(2)};
  KernelMetadata->addOperand(MDNode::get(Ctx, NewMetadata));

  // Create alloca of zeros for the implicit offset in the original func.
  BasicBlock *EntryBlock = &Func->getEntryBlock();
  IRBuilder<> Builder(EntryBlock, EntryBlock->getFirstInsertionPt());
  Type *ImplicitOffsetType =
      ArrayType::get(Type::getInt32Ty(M.getContext()), 3);
  AllocaInst *ImplicitOffset =
      Builder.CreateAlloca(ImplicitOffsetType, TargetAS);
  uint64_t AllocByteSize =
      ImplicitOffset->getAllocationSizeInBits(M.getDataLayout()).value() / 8;
  CallInst *MemsetCall =
      Builder.CreateMemSet(ImplicitOffset, Builder.getInt8(0), AllocByteSize,
                           ImplicitOffset->getAlign());
  MemsetCall->addParamAttr(0, Attribute::NonNull);
  MemsetCall->addDereferenceableParamAttr(0, AllocByteSize);
  ProcessedFunctions[Func] = Builder.CreateConstInBoundsGEP2_32(
      ImplicitOffsetType, ImplicitOffset, 0, 0);
}

void GlobalOffsetPass::addImplicitParameterToCallers(
    Module &M, Value *Callee, Function *CalleeWithImplicitParam) {

  // Make sure that all entry point callers are processed.
  SmallVector<User *, 8> Users{Callee->users()};
  for (User *U : Users) {
    auto *Call = dyn_cast<CallInst>(U);
    if (!Call)
      continue;

    Function *Caller = Call->getFunction();
    if (EntryPointMetadata.count(Caller) != 0) {
      processKernelEntryPoint(Caller);
    }
  }

  // User collection may have changed, so we reinitialize it.
  Users = SmallVector<User *, 8>{Callee->users()};
  for (User *U : Users) {
    auto *CallToOld = dyn_cast<CallInst>(U);
    if (!CallToOld)
      return;

    auto *Caller = CallToOld->getFunction();

    // Determine if `Caller` needs processed or if this is another callsite
    // from an already-processed function.
    Function *NewFunc;
    Value *ImplicitOffset = ProcessedFunctions[Caller];
    bool AlreadyProcessed = ImplicitOffset != nullptr;
    if (AlreadyProcessed) {
      NewFunc = Caller;
    } else {
      std::tie(NewFunc, ImplicitOffset) =
          addOffsetArgumentToFunction(M, Caller);
    }

    if (!CalleeWithImplicitParam) {
      // Replace intrinsic call with parameter.
      CallToOld->replaceAllUsesWith(ImplicitOffset);
    } else {
      // Build up a list of arguments to call the modified function using.
      SmallVector<Value *, 8> ImplicitOffsets;
      for (Use &U : CallToOld->args()) {
        ImplicitOffsets.push_back(U);
      }
      ImplicitOffsets.push_back(ImplicitOffset);

      // Replace call to other function (which now has a new parameter),
      // with a call including the new parameter to that same function.
      auto *NewCallInst = CallInst::Create(
          /* Ty= */ CalleeWithImplicitParam->getFunctionType(),
          /* Func= */ CalleeWithImplicitParam,
          /* Args= */ ImplicitOffsets,
          /* NameStr= */ Twine(),
          /* InsertBefore= */ CallToOld);
      NewCallInst->setTailCallKind(CallToOld->getTailCallKind());
      NewCallInst->copyMetadata(*CallToOld);
      CallToOld->replaceAllUsesWith(NewCallInst);

      if (CallToOld->hasName()) {
        NewCallInst->takeName(CallToOld);
      }
    }

    // Remove the caller now that it has been replaced.
    CallToOld->eraseFromParent();

    if (AlreadyProcessed)
      continue;

    // Process callers of the old function.
    addImplicitParameterToCallers(M, Caller, NewFunc);

    // Now that the old function is dead, delete it.
    Caller->dropAllReferences();
    Caller->eraseFromParent();
  }
}

std::pair<Function *, Value *> GlobalOffsetPass::addOffsetArgumentToFunction(
    Module &M, Function *Func, Type *ImplicitArgumentType, bool KeepOriginal) {
  FunctionType *FuncTy = Func->getFunctionType();
  const AttributeList &FuncAttrs = Func->getAttributes();
  ImplicitArgumentType =
      ImplicitArgumentType ? ImplicitArgumentType : ImplicitOffsetPtrType;

  // Construct an argument list containing all of the previous arguments.
  SmallVector<Type *, 8> Arguments;
  SmallVector<AttributeSet, 8> ArgumentAttributes;
  for (const auto &I : enumerate(Func->args())) {
    Arguments.push_back(I.value().getType());
    ArgumentAttributes.push_back(FuncAttrs.getParamAttrs(I.index()));
  }

  // Add the offset argument. Must be the same type as returned by
  // `llvm.{amdgcn|nvvm}.implicit.offset`.
  Arguments.push_back(ImplicitArgumentType);
  ArgumentAttributes.push_back(AttributeSet());

  // Build the new function.
  AttributeList NAttrs =
      AttributeList::get(Func->getContext(), FuncAttrs.getFnAttrs(),
                         FuncAttrs.getRetAttrs(), ArgumentAttributes);
  assert(!FuncTy->isVarArg() && "Variadic arguments prohibited in SYCL");
  FunctionType *NewFuncTy =
      FunctionType::get(FuncTy->getReturnType(), Arguments, FuncTy->isVarArg());

  Function *NewFunc =
      Function::Create(NewFuncTy, Func->getLinkage(), Func->getAddressSpace());

  // Keep original function ordering.
  M.getFunctionList().insertAfter(Func->getIterator(), NewFunc);

  Value *ImplicitOffset = nullptr;
  bool ImplicitOffsetAllocaInserted = false;
  if (KeepOriginal) {
    // TODO: Are there better naming alternatives that allow for unmangling?
    NewFunc->setName(Func->getName() + "_with_offset");

    ValueToValueMapTy VMap;
    for (Function::arg_iterator FuncArg = Func->arg_begin(),
                                FuncEnd = Func->arg_end(),
                                NewFuncArg = NewFunc->arg_begin();
         FuncArg != FuncEnd; ++FuncArg, ++NewFuncArg) {
      VMap[FuncArg] = NewFuncArg;
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewFunc, Func, VMap,
                      CloneFunctionChangeType::GlobalChanges, Returns);
    // In order to keep the signatures of functions called by the kernel
    // unified, the pass has to copy global offset to an array allocated in
    // addrspace(3). This is done as kernels can't allocate and fill the
    // array in constant address space, which would be required for the case
    // with no global offset.
    if (AT == ArchType::AMDHSA) {
      BasicBlock *EntryBlock = &NewFunc->getEntryBlock();
      IRBuilder<> Builder(EntryBlock, EntryBlock->getFirstInsertionPt());
      Type *ImplicitOffsetType =
          ArrayType::get(Type::getInt32Ty(M.getContext()), 3);
      Value *OrigImplicitOffset = std::prev(NewFunc->arg_end());
      AllocaInst *ImplicitOffsetAlloca =
          Builder.CreateAlloca(ImplicitOffsetType, TargetAS);
      auto DL = M.getDataLayout();
      uint64_t AllocByteSize =
          ImplicitOffsetAlloca->getAllocationSizeInBits(DL).value() / 8;
      // After AMD's kernel arg lowering pass runs the accesses to arguments
      // are replaced with uses of kernarg.segment.ptr which is in
      // addrspace(4), cast implicit offset arg to constant memory so the
      // memcpy is issued into a correct address space.
      auto OrigImplicitOffsetAS4 = Builder.CreateAddrSpaceCast(
          OrigImplicitOffset, Type::getInt8Ty(M.getContext())->getPointerTo(4));
      Builder.CreateMemCpy(
          ImplicitOffsetAlloca, ImplicitOffsetAlloca->getAlign(),
          OrigImplicitOffsetAS4, OrigImplicitOffsetAS4->getPointerAlignment(DL),
          AllocByteSize);
      ImplicitOffset = ImplicitOffsetAlloca;
      ImplicitArgumentType = ImplicitOffset->getType();
      ImplicitOffsetAllocaInserted = true;
    } else {
      ImplicitOffset = std::prev(NewFunc->arg_end());
    }
  } else {
    NewFunc->copyAttributesFrom(Func);
    NewFunc->setComdat(Func->getComdat());
    NewFunc->setAttributes(NAttrs);
    NewFunc->takeName(Func);

    // Splice the body of the old function right into the new function.
    NewFunc->splice(NewFunc->begin(), Func);

    for (Function::arg_iterator FuncArg = Func->arg_begin(),
                                FuncEnd = Func->arg_end(),
                                NewFuncArg = NewFunc->arg_begin();
         FuncArg != FuncEnd; ++FuncArg, ++NewFuncArg) {
      FuncArg->replaceAllUsesWith(NewFuncArg);
    }

    // Clone metadata of the old function, including debug info descriptor.
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    Func->getAllMetadata(MDs);
    for (auto MD : MDs)
      NewFunc->addMetadata(MD.first, *MD.second);

    ImplicitOffset = std::prev(NewFunc->arg_end());
  }
  assert(ImplicitOffset && "Value of implicit offset must be set.");

  // Add bitcast to match the return type of the intrinsic if needed.
  if (ImplicitArgumentType != ImplicitOffsetPtrType) {
    BasicBlock *EntryBlock = &NewFunc->getEntryBlock();
    // Make sure bitcast is inserted after alloca, if present.
    BasicBlock::iterator InsertionPt =
        ImplicitOffsetAllocaInserted
            ? std::next(cast<AllocaInst>(ImplicitOffset)->getIterator())
            : EntryBlock->getFirstInsertionPt();
    IRBuilder<> Builder(EntryBlock, InsertionPt);
    ImplicitOffset = Builder.CreateBitCast(
        ImplicitOffset,
        Type::getInt32Ty(M.getContext())->getPointerTo(TargetAS));
  }

  ProcessedFunctions[NewFunc] = ImplicitOffset;

  // Return the new function and the offset argument.
  return {NewFunc, ImplicitOffset};
}

DenseMap<Function *, MDNode *> GlobalOffsetPass::generateKernelMDNodeMap(
    Module &M, SmallVectorImpl<KernelPayload> &KernelPayloads) {
  SmallPtrSet<GlobalValue *, 8u> Used;
  SmallVector<GlobalValue *, 4> Vec;
  collectUsedGlobalVariables(M, Vec, /*CompilerUsed=*/false);
  collectUsedGlobalVariables(M, Vec, /*CompilerUsed=*/true);
  Used = {Vec.begin(), Vec.end()};

  auto HasUseOtherThanLLVMUsed = [&Used](GlobalValue *GV) {
    if (GV->use_empty())
      return false;
    return !GV->hasOneUse() || !Used.count(GV);
  };

  DenseMap<Function *, MDNode *> EntryPointMetadata;
  for (auto &KP : KernelPayloads) {
    if (HasUseOtherThanLLVMUsed(KP.Kernel))
      llvm_unreachable("Kernel entry point can't have uses.");

    EntryPointMetadata[KP.Kernel] = KP.MD;
  }

  return EntryPointMetadata;
}
