//===------ MutatePrintfAddrspace.cpp - SYCL printf AS mutation Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which detects non-constant address space
// literals usage for the first argument of SYCL experimental printf
// function, and moves the string literal to constant address
// space. This a temporary solution for printf's support of generic
// address space literals; the pass should be dropped once SYCL device
// backends learn to handle the generic address-spaced argument properly.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/MutatePrintfAddrspace.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace {
// Wrapper for the pass to make it working with the old pass manager
class SYCLMutatePrintfAddrspaceLegacyPass : public ModulePass {
public:
  static char ID;
  SYCLMutatePrintfAddrspaceLegacyPass() : ModulePass(ID) {
    initializeSYCLMutatePrintfAddrspaceLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  // run the SYCLMutatePrintfAddrspace pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLMutatePrintfAddrspacePass Impl;
};

struct CallReplacer;
class AddrspaceReplacer {
public:
  static constexpr unsigned ConstantAddrspaceID = 2;

  AddrspaceReplacer() = delete;
  AddrspaceReplacer(Module *M) : M(M) {
    Type *Int8Ty = Type::getInt8Ty(M->getContext());
    CASLiteralType = PointerType::get(Int8Ty, ConstantAddrspaceID);
  }
  ~AddrspaceReplacer() {
    for (Function *F : FunctionsToDrop)
      F->eraseFromParent();
  }
  size_t getReplacedCallsCount() { return ReplacedCallsCount; }

  void runOnFunctionDeclaration(Function *F);

private:
  Module *M;
  PointerType *CASLiteralType;
  size_t ReplacedCallsCount = 0;
  /// If the variadic version gets picked during FE compilation, we'll only have
  /// 1 function to replace. However, unique declarations are emitted for each
  /// of the non-variadic (variadic template) calls.
  SmallVector<Function *, 8> FunctionsToDrop;

  Function *getCASPrintfFunction(Function *GenericASPrintfFunc);
  Constant *getCASLiteral(GlobalVariable *GenericASLiteral);
};
} // namespace

char SYCLMutatePrintfAddrspaceLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLMutatePrintfAddrspaceLegacyPass,
                "SYCLMutatePrintfAddrspace",
                "Move SYCL printf literal arguments to constant address space",
                false, false)

// Public interface to the SYCLMutatePrintfAddrspacePass.
ModulePass *llvm::createSYCLMutatePrintfAddrspaceLegacyPass() {
  return new SYCLMutatePrintfAddrspaceLegacyPass();
}

PreservedAnalyses
SYCLMutatePrintfAddrspacePass::run(Module &M, ModuleAnalysisManager &MAM) {
  AddrspaceReplacer ASReplacer(&M);
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    if (!F.getName().startswith("_Z18__spirv_ocl_printf"))
      continue;
    ASReplacer.runOnFunctionDeclaration(&F);
  }
  return ASReplacer.getReplacedCallsCount() ? PreservedAnalyses::all()
                                            : PreservedAnalyses::none();
}

/// Helper implementations
namespace {

/// Encapsulates the update of CallInst's literal argument.
struct CallReplacer {
  CallReplacer(CallInst *CI, Value *CASArg) : CI(CI), CASArg(CASArg) {}
  void replaceWithFunction(Function *CASPrintf) {
    CI->setCalledFunction(CASPrintf);
    Value *ArgToSet = CASArg;
    if (auto *Const = dyn_cast<Constant>(CASArg)) {
      // In case there's a misalignment between the updated function type and
      // the constant literal type, create a constant pointer cast so as to
      // duck module verifier complaints.
      Type *ParamType = CASPrintf->getFunctionType()->getParamType(0);
      if (Const->getType() != ParamType)
        ArgToSet = ConstantExpr::getPointerCast(Const, ParamType);
    }
    CI->setArgOperand(0, ArgToSet);
  }

private:
  CallInst *CI;
  Value *CASArg;
};

/// The function's effect is similar to V->stripPointerCastsAndAliases(), but
/// also strips load/store aliases.
Value *stripToMemorySource(Value *V) {
  Value *MemoryAccess = V;
  if (auto *LI = dyn_cast<LoadInst>(MemoryAccess)) {
    Value *LoadSource = LI->getPointerOperand();
    auto *Store = cast<StoreInst>(*llvm::find_if(
        LoadSource->users(), [](User *U) { return isa<StoreInst>(U); }));
    MemoryAccess = Store->getValueOperand();
  }
  return MemoryAccess->stripPointerCastsAndAliases();
}

void AddrspaceReplacer::runOnFunctionDeclaration(Function *F) {
  auto *LiteralType = F->getArg(0)->getType();
  if (LiteralType->getPointerAddressSpace() == ConstantAddrspaceID)
    // No need to replace the literal type and its printf users
    return;
  SmallVector<CallReplacer, 16> CallsToReplace;
  for (User *U : F->users()) {
    if (!isa<CallInst>(U))
      continue;
    auto *CI = cast<CallInst>(U);

    /// This key algorithm reaches the global string used as an argument to a
    /// __spirv_ocl_printf call. It then generates a constant AS copy of that
    /// global (or gets an existing one). For the return value, the call
    /// instruction is paired with its future constant addrspace string
    /// argument.
    Value *Stripped = stripToMemorySource(CI->getArgOperand(0));
    if (auto *Literal = dyn_cast<GlobalVariable>(Stripped))
      CallsToReplace.emplace_back(CI, getCASLiteral(Literal));
    else if (auto *Arg = dyn_cast<Argument>(Stripped)) {
      // The global literal is passed to __spirv_ocl_printf via a wrapper
      // function argument. We'll update the wrapper calls to use the builtin
      // function directly instead.
      Function *WrapperFunc = Arg->getParent();
      for (User *WrapperU : WrapperFunc->users()) {
        auto *WrapperCI = cast<CallInst>(WrapperU);
        Value *StrippedArg = stripToMemorySource(WrapperCI->getArgOperand(0));
        // We're only expecting 1 level of wrappers, so cast unconditionally
        auto *Literal = cast<GlobalVariable>(StrippedArg);
        CallsToReplace.emplace_back(WrapperCI, getCASLiteral(Literal));
      }
      // We're certain that the wrapper won't have any uses, since we've just
      // marked all its calls for replacement with __spirv_ocl_printf.
      FunctionsToDrop.emplace_back(WrapperFunc);
      // Similar certainty for the generic AS version of __spirv_ocl_printf
      // itself - we've determined it only gets called inside the
      // soon-to-be-removed wrapper.
      assert(F->hasOneUse() && "Unexpected __spirv_ocl_printf call outside of "
                               "SYCL wrapper function");
      FunctionsToDrop.emplace_back(F);
    } else
      llvm_unreachable(
          "Unexpected literal operand type for device-side printf");
  }
  Function *CASPrintfFunc = getCASPrintfFunction(F);
  for (CallReplacer &CR : CallsToReplace) {
    CR.replaceWithFunction(CASPrintfFunc);
    ++ReplacedCallsCount;
  }
  if (F->hasNUses(0))
    FunctionsToDrop.emplace_back(F);
}

/// Get the constant addrspace version of the __spirv_ocl_printf declaration, or
/// generate it if the IR module doesn't have it yet. Also make it variadic so
/// that it could replace all non-variadic generic AS versions.
Function *
AddrspaceReplacer::getCASPrintfFunction(Function *GenericASPrintfFunc) {
  auto *CASPrintfFuncTy =
      FunctionType::get(GenericASPrintfFunc->getReturnType(), CASLiteralType,
                        /*isVarArg=*/true);
  FunctionCallee CASPrintfFunc =
      M->getOrInsertFunction("_Z18__spirv_ocl_printfPU3AS2Kcz", CASPrintfFuncTy,
                             GenericASPrintfFunc->getAttributes());
  auto *Callee = cast<Function>(CASPrintfFunc.getCallee());
  Callee->setCallingConv(CallingConv::SPIR_FUNC);
  Callee->setDSOLocal(true);
  return Callee;
}

/// Generate the constant addrspace version of the generic addrspace-residing
/// global string. If one exists already, get it from the module.
Constant *AddrspaceReplacer::getCASLiteral(GlobalVariable *GenericASLiteral) {
  // Appending the stable suffix ensures that only one CAS copy is made for each
  // string. In case of the matching name, llvm::Module APIs will ensure that
  // the existing global is returned.
  std::string CASLiteralName = GenericASLiteral->getName().str() + "._AS2";
  IRBuilder<> Builder(M->getContext());
  if (GlobalVariable *ExistingGlobal =
          M->getGlobalVariable(CASLiteralName, /*AllowInternal=*/true))
    return ExistingGlobal;

  StringRef LiteralValue;
  getConstantStringInfo(GenericASLiteral, LiteralValue);
  GlobalVariable *Res = Builder.CreateGlobalString(LiteralValue, CASLiteralName,
                                                   ConstantAddrspaceID, M);
  Res->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  Res->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  return Res;
}
} // namespace
