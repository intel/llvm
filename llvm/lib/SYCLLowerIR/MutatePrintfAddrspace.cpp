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

static constexpr unsigned ConstantAddrspaceID = 2;
// If the variadic version gets picked during FE compilation, we'll only have
// 1 function to replace. However, unique declarations are emitted for each
// of the non-variadic (variadic template) calls.
using FunctionVecTy = SmallVector<Function *, 8>;

Function *getCASPrintfFunction(Module &M, PointerType *CASLiteralType);
size_t setFuncCallsOntoCASPrintf(Function *F, Function *CASPrintfFunc,
                                 FunctionVecTy &FunctionsToDrop);
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
  Type *Int8Type = Type::getInt8Ty(M.getContext());
  auto *CASLiteralType = PointerType::get(Int8Type, ConstantAddrspaceID);
  Function *CASPrintfFunc = getCASPrintfFunction(M, CASLiteralType);

  FunctionVecTy FunctionsToDrop;
  bool ModuleChanged = false;
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    if (!F.getName().startswith("_Z18__spirv_ocl_printf"))
      continue;
    if (F.getArg(0)->getType() == CASLiteralType)
      // No need to replace the literal type and its printf users
      continue;
    ModuleChanged |=
        setFuncCallsOntoCASPrintf(&F, CASPrintfFunc, FunctionsToDrop);
  }
  for (Function *F : FunctionsToDrop)
    F->eraseFromParent();

  return ModuleChanged ? PreservedAnalyses::all() : PreservedAnalyses::none();
}

/// Helper implementations
namespace {

/// Get the constant addrspace version of the __spirv_ocl_printf declaration,
/// or generate it if the IR module doesn't have it yet. Also make it
/// variadic so that it could replace all non-variadic generic AS versions.
Function *getCASPrintfFunction(Module &M, PointerType *CASLiteralType) {
  Type *Int32Type = Type::getInt32Ty(M.getContext());
  auto *CASPrintfFuncTy = FunctionType::get(Int32Type, CASLiteralType,
                                            /*isVarArg=*/true);
  // extern int __spirv_ocl_printf(
  //                const __attribute__((opencl_constant)) char *Format, ...)
  FunctionCallee CASPrintfFuncCallee =
      M.getOrInsertFunction("_Z18__spirv_ocl_printfPU3AS2Kcz", CASPrintfFuncTy);
  auto *CASPrintfFunc = cast<Function>(CASPrintfFuncCallee.getCallee());
  CASPrintfFunc->setCallingConv(CallingConv::SPIR_FUNC);
  CASPrintfFunc->setDSOLocal(true);
  return CASPrintfFunc;
}

/// Generate the constant addrspace version of the generic addrspace-residing
/// global string. If one exists already, get it from the module.
Constant *getCASLiteral(GlobalVariable *GenericASLiteral) {
  Module *M = GenericASLiteral->getParent();
  // Appending the stable suffix ensures that only one CAS copy is made for each
  // string. In case of the matching name, llvm::Module APIs will ensure that
  // the existing global is returned.
  std::string CASLiteralName = GenericASLiteral->getName().str() + "._AS2";
  if (GlobalVariable *ExistingGlobal =
          M->getGlobalVariable(CASLiteralName, /*AllowInternal=*/true))
    return ExistingGlobal;

  StringRef LiteralValue;
  getConstantStringInfo(GenericASLiteral, LiteralValue);
  IRBuilder<> Builder(M->getContext());
  GlobalVariable *Res = Builder.CreateGlobalString(LiteralValue, CASLiteralName,
                                                   ConstantAddrspaceID, M);
  Res->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  Res->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  return Res;
}

/// Encapsulates the update of CallInst's literal argument.
void setCallArgOntoCASPrintf(CallInst *CI, Constant *CASArg,
                             Function *CASPrintfFunc) {
  CI->setCalledFunction(CASPrintfFunc);
  auto *Const = CASArg;
  // In case there's a misalignment between the updated function type and
  // the constant literal type, create a constant pointer cast so as to
  // duck module verifier complaints.
  Type *ParamType = CASPrintfFunc->getFunctionType()->getParamType(0);
  if (Const->getType() != ParamType)
    Const = ConstantExpr::getPointerCast(Const, ParamType);
  CI->setArgOperand(0, Const);
}

/// The function's effect is similar to V->stripPointerCastsAndAliases(), but
/// also strips load/store aliases.
/// NB: This function can only operate on simple CFG, where load/store pairs
/// leading to the global variable are merely a consequence of low optimization
/// level. Re-using it for complex CFG with arbitrary memory paths is definitely
/// not recommended.
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

void emitError(Function *PrintfInstance, CallInst *PrintfCall,
               StringRef RecommendationToUser = "") {
  std::string ErrorMsg =
      std::string("experimental::printf requires format string to reside "
                  "in constant "
                  "address space. The compiler wasn't able to "
                  "automatically convert "
                  "your format string into constant address space when "
                  "processing builtin ") +
      PrintfInstance->getName().str() + " called in function " +
      PrintfCall->getFunction()->getName().str() + ".\n" +
      RecommendationToUser.str();
  PrintfInstance->getContext().emitError(PrintfCall, ErrorMsg);
}

/// This routine goes over CallInst users of F, resetting the called function
/// to CASPrintfFunc and generating/retracting constant addrspace format
/// strings to use as operands of the mutated calls.
size_t setFuncCallsOntoCASPrintf(Function *F, Function *CASPrintfFunc,
                                 FunctionVecTy &FunctionsToDrop) {
  size_t MutatedCallsCount = 0;
  SmallVector<std::pair<CallInst *, Constant *>, 16> CallsToMutate;
  FunctionVecTy WrapperFunctionsToDrop;
  for (User *U : F->users()) {
    if (!isa<CallInst>(U))
      continue;
    auto *CI = cast<CallInst>(U);

    // This key algorithm reaches the global string used as an argument to a
    // __spirv_ocl_printf call. It then generates a constant AS copy of that
    // global (or gets an existing one). For the return value, the call
    // instruction is paired with its future constant addrspace string
    // argument.
    Value *Stripped = stripToMemorySource(CI->getArgOperand(0));
    if (auto *Literal = dyn_cast<GlobalVariable>(Stripped))
      CallsToMutate.emplace_back(CI, getCASLiteral(Literal));
    else if (auto *Arg = dyn_cast<Argument>(Stripped)) {
      // The global literal is passed to __spirv_ocl_printf via a wrapper
      // function argument. We'll update the wrapper calls to use the builtin
      // function directly instead.
      Function *WrapperFunc = Arg->getParent();
      std::string BadWrapperErrorMsg =
          "Consider simplifying the code by "
          "passing format strings directly into experimental::printf calls, "
          "avoiding indirection via wrapper function arguments.";
      if (!WrapperFunc->getName().contains("6oneapi12experimental6printf")) {
        emitError(F, CI, BadWrapperErrorMsg);
        return 0;
      }
      for (User *WrapperU : WrapperFunc->users()) {
        auto *WrapperCI = cast<CallInst>(WrapperU);
        Value *StrippedArg = stripToMemorySource(WrapperCI->getArgOperand(0));
        auto *Literal = dyn_cast<GlobalVariable>(StrippedArg);
        // We only expect 1 level of wrappers
        if (!Literal) {
          emitError(WrapperFunc, WrapperCI, BadWrapperErrorMsg);
          return 0;
        }
        CallsToMutate.emplace_back(WrapperCI, getCASLiteral(Literal));
      }
      // We're certain that the wrapper won't have any uses, since we've just
      // marked all its calls for replacement with __spirv_ocl_printf.
      WrapperFunctionsToDrop.emplace_back(WrapperFunc);
      // __spirv_ocl_printf itself only gets called inside the
      // soon-to-be-removed wrappers and will be marked for removal once these
      // are removed. The builtin may only have n>1 uses in case it's variadic -
      // in that scenario, all wrapper instances will be referencing it.
      assert((F->hasOneUse() || F->isVarArg()) &&
             "Unexpected __spirv_ocl_printf call outside of "
             "SYCL wrapper function");
    } else {
      emitError(
          F, CI,
          "Make sure each format string literal is "
          "known at compile time or use OpenCL constant address space literals "
          "for device-side printf calls.");
      return 0;
    }
  }
  for (auto &CallConstantPair : CallsToMutate) {
    setCallArgOntoCASPrintf(CallConstantPair.first, CallConstantPair.second,
                            CASPrintfFunc);
    ++MutatedCallsCount;
  }
  for (Function *WF : WrapperFunctionsToDrop)
    WF->eraseFromParent();
  if (F->hasNUses(0))
    FunctionsToDrop.emplace_back(F);
  return MutatedCallsCount;
}
} // namespace
