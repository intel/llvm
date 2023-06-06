//===---- LowerESIMDKernelAttrs - lower __esimd_set_kernel_attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes passing and returning esimd::simd objects - in
// non-__regcall functions they are passed by pointer (glitch of the spir_func
// calling convention). The IR - a function return value and formal parameters,
// when eligible, along with all callers - are transformed so that the simd
// objects are passed by value. The pass assumes:
// - Opaque Pointers are turned on.
// - It is run on whole program (i.e. from the post-link tool), and all callers
//   of the optimized funtion are a part of the module.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

#include "llvm/GenXIntrinsics/GenXIntrinsics.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <variant>

#define DEBUG_TYPE "ESIMDOptimizeVecArgCallConv"

using namespace llvm;

static bool isESIMDVectorTy(Type *T) {
  return T->isVectorTy() ||
         esimd::getVectorTyOrNull(dyn_cast<StructType>(T)) != nullptr;
}

static Type *getVectorTypeOrNull(Type *T) {
  if (!T || T->isVectorTy()) {
    return T;
  }
  Type *Res = esimd::getVectorTyOrNull(dyn_cast<StructType>(T));
  return Res;
}

// Checks types equivalence for the purpose if this optimization.
// Thin struct wrapper types around a vector type are equivalent between them
// and the vector type.
static bool eq(Type *T1, Type *T2) {
  if (T1 == T2) {
    return true;
  }
  if (Type *T1V = getVectorTypeOrNull(T1)) {
    return T1V == getVectorTypeOrNull(T2);
  } else {
    return false;
  }
}

using NonMemUseHandlerT = std::function<bool(const Use *)>;

static Type *
getMemTypeIfSameAddressLoadsStores(SmallPtrSetImpl<const Use *> &Uses,
                                   bool &LoadMet, bool &StoreMet,
                                   NonMemUseHandlerT *NonMemUseF = nullptr) {
  LoadMet = false;
  StoreMet = false;
  Type *ContentT = nullptr;

  if (Uses.size() == 0) {
    return nullptr;
  }
  Value *Addr = sycl::utils::stripCastsAndZeroGEPs((*Uses.begin())->get());

  for (const auto *UU : Uses) {
    const User *U = UU->getUser();

    if (const auto *LI = dyn_cast<LoadInst>(U)) {
      if (!ContentT) {
        ContentT = LI->getType();
        LoadMet = 1;
        continue;
      } else if (!eq(ContentT, LI->getType())) {
        return nullptr;
      }
    }

    if (const auto *SI = dyn_cast<StoreInst>(U)) {
      if (sycl::utils::stripCastsAndZeroGEPs(SI->getPointerOperand()) != Addr) {
        // the pointer escapes into memory
        return nullptr;
      }
      if (!ContentT) {
        ContentT = SI->getValueOperand()->getType();
        StoreMet = 1;
        continue;
      } else if (!eq(ContentT, SI->getValueOperand()->getType())) {
        return nullptr;
      }
    }
    if (!NonMemUseF || !(*NonMemUseF)(UU)) {
      // bail out early if there is no "non-memory use" handler or the handler
      // requests bailout
      return nullptr;
    }
  }
  return getVectorTypeOrNull(ContentT);
}

static bool isSretParam(const Argument &P) {
  return P.getParamStructRetType() != nullptr;
}

// See if given formal parameter (which can be an 'sret' one) is optimizeable,
// and return its value type if is or nullptr otherwise.
Type *getPointedToTypeIfOptimizeable(const Argument &FormalParam) {
  if (!FormalParam.getType()->isPointerTy()) {
    return nullptr;
  }
  Type *ContentT = nullptr;
  const bool IsSret = isSretParam(FormalParam);

  // Check if internal optimization criteria is met for the parameter (dataflow
  // through the pointer value represented by the parameter within the callee IR
  // matches expectation) and get the pointed-to memory type.
  // The expected IR is as follows (%ret is the return value, %ptr_param is a
  // formal parameter):
  //   void @the_func(ptr sret(<N x T>) %ret, ptr %ptr_param) {
  //     ...
  //     %param1 = load <N1 x T1> %ptr_param ...
  //     <use %param1>
  //     ...
  //     %param2 = load <N1 x T1> %ptr_param ...
  //     <use %param2>
  //     ...
  //     store <N x T> %some_val1, %ret ; e.g. true branch
  //     ...
  //     store <N x T> %some_val2, %ret ; e.g. false branch
  //     ...
  //     ret void
  //   }
  // It is subsequently transformed to ([+] - added line, [c]-changed line):
  //   [c] <N x T> @the_func(<N1 x T1> %param) {
  //     [+] %ptr_param = alloca <N1 x T1>
  //     [+] store <N1 x T1> %ptr_param, %param
  //     [+] %ret = alloca <N x T>
  //     ...
  //     %param1 = load <N1 x T1> %ptr_param ...
  //     ...
  //     <use %param1>
  //     ...
  //     %param2 = load <N1 x T1> %ptr_param ...
  //     <use %param2>
  //     ...
  //     store <N x T> %some_val1, %ret ; e.g. true branch
  //     ...
  //     store <N x T> %some_val2, %ret ; e.g. false branch
  //     ...
  //     [+] %ret_val = load <N x T> %ret
  //     [c] ret <N x T> %ret_val
  //   }
  {
    SmallPtrSet<const Use *, 4> Uses;
    sycl::utils::collectUsesLookThroughCastsAndZeroGEPs(&FormalParam, Uses);
    bool LoadMet = 0;
    bool StoreMet = 0;
    ContentT = getMemTypeIfSameAddressLoadsStores(Uses, LoadMet, StoreMet);
    const bool NonOptimizeableParam = !IsSret && (StoreMet || !LoadMet);

    if (IsSret) {
      Type *SretVecT = getVectorTypeOrNull(FormalParam.getParamStructRetType());

      if (!ContentT) {
        // Can happen when sret param is a "fall through" - return value is
        // assigned in callees.
        ContentT = SretVecT;
      } else {
        if (ContentT != SretVecT) {
          return nullptr;
        }
      }
    }
    if (!ContentT || NonOptimizeableParam) {
      return nullptr;
    }
  }
  // Check if external criteria is met for the param at each call site
  // (data flow in callers' IR match expectation). The following IR is expected
  // in the callers of '@the_func', in which return value and its only parameter
  // are optimized:
  //   %ptr_param = alloca <N x T>
  //   %ret = alloca <N1 x T1>
  //   ...
  //   store <N x T> %some_val, %ptr_param
  //   ...
  //   call void @the_func(%ret, %ptr_param)
  //   ...
  //   %ret_val = load <N1 x T1> %ret
  //   ...
  //   <use %ret_val>
  // Subsequently, this IR is optimized to ([+] - added line, [c]-changed line):
  //   %ptr_param = alloca <N x T>
  //   %ret = alloca <N1 x T1>
  //   ...
  //   store <N x T> %some_val, %ptr_param
  //   ...
  //   [+] %pointed_to_val = load <N x T> %ptr_param
  //   [c] call %ret_val = call <N1 x T1> @the_func(%pointed_to_val)
  //   [+] store <N1 x T1> %ret_val, %ret
  //   ...
  //   %ret_val = load <N1 x T1> %ret
  //   ...
  //   <use %ret_val>
  const Function *F = FormalParam.getParent();

  for (auto *U : F->users()) {
    auto *Call = dyn_cast<CallInst>(U);

    if (!Call || (Call->getCalledFunction() != F)) {
      return nullptr;
    }
    Value *ActualParam = sycl::utils::stripCastsAndZeroGEPs(
        Call->getArgOperand(FormalParam.getArgNo()));

    if (!IsSret && !isa<AllocaInst>(ActualParam)) {
      return nullptr;
    }
    SmallPtrSet<const Use *, 4> Uses;
    sycl::utils::collectUsesLookThroughCastsAndZeroGEPs(ActualParam, Uses);
    bool LoadMet = 0;
    bool StoreMet = 0;

    // Handler for the case when non-memory access use instruction is met.
    NonMemUseHandlerT NonMemUseMetF = [&](const Use *AUse) {
      if (auto CI = dyn_cast<CallInst>(AUse->getUser())) {
        // if not equal, alloca escapes to some other function
        return CI->getCalledFunction() == F;
      }
      // Returning false forces early bailout in the caller.
      return false;
    };
    Type *T = getMemTypeIfSameAddressLoadsStores(Uses, LoadMet, StoreMet,
                                                 &NonMemUseMetF);
    const bool NonOptimizeableParam = !IsSret && !StoreMet;

    if ((T && (T != ContentT)) || NonOptimizeableParam) {
      // T != ContentT means either T is null or pointed-to type of the actual
      // param is different from the pointed-to type of the formal param
      return nullptr;
    }
  }
  return ContentT;
}

class FormalParamInfo {
  Argument &P;
  Type *OptimizedType;

public:
  FormalParamInfo(Argument &FormalParam)
      : P(FormalParam), OptimizedType(nullptr) {
    OptimizedType = getPointedToTypeIfOptimizeable(P);
  }

  const Argument &getFormalParam() const { return P; }

  bool isSret() const { return ::isSretParam(P); }

  Type *getOptimizedType() const { return OptimizedType; }

  bool canOptimize() const { return getOptimizedType() != nullptr; }
};

static unsigned oldArgNo2NewArgNo(unsigned OldArgNo, int SretInd) {
  unsigned NewArgNo =
      (SretInd < 0) || ((int)OldArgNo < SretInd) ? OldArgNo : OldArgNo - 1;
  return NewArgNo;
}

// Transforms
//   void @the_func(ptr sret(<N x T>) %ret, ptr %ptr_param) {
//     ...
//     ret void
//   }
// to ([+] - added line, [c]-changed line):
//   [c] <N1 x T1> @the_func(<N x T> %param) {
//     [+] %ptr_param = alloca <N x T>
//     [+] store <N x T> %ptr_param, %param
//     [+] %ret = alloca <N1 x T1>
//     ...
//     [+] %ret_val = load <N1 x T1> %ret
//     [c] ret <N1 x T1> %ret_val
//   }
static Function *
optimizeFunction(Function *OldF,
                 const SmallVectorImpl<FormalParamInfo> &OptimizeableParams,
                 const SmallVectorImpl<Type *> &NewParamTs) {
  // 1. Find optimizeable 'sret' param index within the param info array.
  int SretInd = -1;
  for (unsigned I = 0; I < OptimizeableParams.size(); ++I) {
    if (OptimizeableParams[I].isSret()) {
      SretInd = I;
      break;
    }
  }
  // 2. Create cloned function declaration.
  // 2.1. Create a new function type.
  FunctionType *OldFT = OldF->getFunctionType();
  Type *NewRetT = SretInd >= 0 ? OptimizeableParams[SretInd].getOptimizedType()
                               : OldFT->getReturnType();
  FunctionType *NewFT = FunctionType::get(NewRetT, NewParamTs, false);
  // 2.2. Create declaration.
  Function *NewF = Function::Create(NewFT, OldF->getLinkage(), OldF->getName(),
                                    OldF->getParent());
  // 3. Create a value map to replace some of the old values.
  ValueToValueMapTy VMap;
  SmallVector<Instruction *, 4> NewInsts;

  // 3.1. Replace each load from the optimizeable ptr parameter with the
  // corresponding new value-typed parameter.
  for (const auto &PI : OptimizeableParams) {
    // Create an alloca for the parameter, and map the original foramal
    // parameter to it.
    Type *T = PI.getOptimizedType();
    const DataLayout &DL = NewF->getParent()->getDataLayout();
    Align Al = DL.getPrefTypeAlign(T);
    unsigned AddrSpace = DL.getAllocaAddrSpace();
    AllocaInst *Alloca = new AllocaInst(T, AddrSpace, 0 /*array size*/, Al);
    NewInsts.push_back(Alloca);
    Instruction *ReplaceInst = Alloca;
    if (auto *ArgPtrType = dyn_cast<PointerType>(PI.getFormalParam().getType());
        ArgPtrType && ArgPtrType->getAddressSpace() != AddrSpace) {
      // If the alloca addrspace and arg addrspace are different,
      // insert a cast.
      ReplaceInst = new AddrSpaceCastInst(Alloca, ArgPtrType);
      NewInsts.push_back(ReplaceInst);
    }
    VMap[OldF->getArg(PI.getFormalParam().getArgNo())] = ReplaceInst;

    if (!PI.isSret()) {
      // Create a store of the new optimized parameter into the alloca to
      // preserve data flow equality to the original.
      unsigned OldArgNo = PI.getFormalParam().getArgNo();
      unsigned NewArgNo = oldArgNo2NewArgNo(OldArgNo, SretInd);
      Instruction *At = nullptr;
      Value *Val = NewF->getArg(NewArgNo);
      StoreInst *St = new StoreInst(Val, Alloca, false, Al, At);
      NewInsts.push_back(St);
    }
  }
  // 3.2. Map unoptimized formal parameters to new ones.
  for (const auto &P : OldF->args()) {
    if (VMap.count(&P) == 0) {
      unsigned NewArgNo = oldArgNo2NewArgNo(P.getArgNo(), SretInd);
      VMap[&P] = NewF->getArg(NewArgNo);
    }
  }
  // 4. Finally, clone the function.
  SmallVector<ReturnInst *, 4> Returns;
  llvm::CloneFunctionInto(NewF, OldF, VMap,
                          CloneFunctionChangeType::LocalChangesOnly, Returns);

  // 5. Fixup returns if this is optimized sret case.
  if (SretInd >= 0) {
    for (ReturnInst *RI : Returns) {
      IRBuilder<> Bld(RI);
      const FormalParamInfo &PI = OptimizeableParams[SretInd];
      Argument *OldP = OldF->getArg(PI.getFormalParam().getArgNo());
      auto *SretPtr = cast<Instruction>(VMap[OldP]);
      if (!isa<AllocaInst>(SretPtr)) {
        auto *AddrSpaceCast = cast<AddrSpaceCastInst>(SretPtr);
        SretPtr = cast<AllocaInst>(AddrSpaceCast->getPointerOperand());
      }
      LoadInst *Ld = Bld.CreateLoad(PI.getOptimizedType(), SretPtr);
      Bld.CreateRet(Ld);
    }
    for (ReturnInst *RI : Returns) {
      RI->eraseFromParent();
    }
  }
  Instruction *At = &*(NewF->getEntryBlock().begin());

  for (unsigned I = 0; I < NewInsts.size(); ++I) {
    NewInsts[I]->insertBefore(At);
  }
  return NewF;
}

// Transform (non-optimizeable actual parameters are not shown)
//   call void @the_func(ptr sret(%simd<T, N>) %sret_ptr, ptr %param_ptr)
// to
//   %param = load <N x T>, ptr %param_ptr
//   %ret = call <N x T> @the_func(<N x T> %param)
//   store <N x T> %ret, ptr %sret_ptr
//
void optimizeCall(CallInst *CI, Function *OptF,
                  SmallVectorImpl<FormalParamInfo> &OptimizeableParams) {
  SmallVector<Value *, 8> NewActualParams;
  unsigned LastActualParamInd = 0;
  int SretInd = -1;
  IRBuilder<> Bld(CI); // insert before CI

  for (unsigned I = 0; I < OptimizeableParams.size(); ++I) {
    const auto &PI = OptimizeableParams[I];
    auto ArgNo = PI.getFormalParam().getArgNo();

    // copy old non-optimized parameters
    auto It0 = CI->arg_begin() + LastActualParamInd;
    auto It1 = CI->arg_begin() + ArgNo;
    std::copy(It0, It1, std::back_inserter(NewActualParams));
    LastActualParamInd = ArgNo + 1;

    if (PI.isSret()) {
      SretInd = I;
      continue; // optimizeable sret is just removed
    }
    // generate optimized replacement - a load from the old actual parameter
    NewActualParams.push_back(
        Bld.CreateLoad(PI.getOptimizedType(), CI->getArgOperand(ArgNo)));
  }
  // Copy remaining unoptimized args if any.
  auto It0 = CI->arg_begin() + LastActualParamInd;
  auto It1 = CI->arg_end();
  std::copy(It0, It1, std::back_inserter(NewActualParams));

  // Create optimized call.
  CallInst *NewCI = Bld.CreateCall(OptF, NewActualParams);
  assert((SretInd == -1) || (CI->getNumUses() == 0));
  NewCI->copyIRFlags(CI);
  NewCI->setCallingConv(CI->getCallingConv());
  NewCI->setTailCall(CI->isTailCall());

  if (SretInd < 0) {
    CI->replaceAllUsesWith(NewCI);
  } else {
    assert(CI->getType()->isVoidTy());
    // Return value was also optimized - store it into memory pointed to by the
    // old sret actual parameter (now alloca instruction).
    unsigned SretActualParamInd =
        OptimizeableParams[SretInd].getFormalParam().getArgNo();
    Bld.CreateStore(NewCI, CI->getArgOperand(SretActualParamInd));
  }
  CI->eraseFromParent();
}

// If function has pointer arguments 'sret' pointer or "by-value" candidate
// pointer argument
static bool processFunction(Function *F) {
  if (esimd::isKernel(*F)) {
    return false;
  }
  // Iterate over formal parameters (return value does not matter, as we are
  // only interested in structure return types, which are always converted to
  // an 'sret' parameter) and
  // - collect data flow for those which can be optimized
  // - collect parameter types of for the new (optimized) function version
  SmallVector<FormalParamInfo, 4> OptimizeableParams;
  SmallVector<Type *, 8> NewParamTs;

  for (auto &Arg : F->args()) {
    FormalParamInfo PI(Arg);

    // We are only interested in esimd::<N x T> types in this pass for now.
    // TODO: extend to other types?
    if (!PI.canOptimize() || !isESIMDVectorTy(PI.getOptimizedType())) {
      // parameter is not optimized - type is not to be changed
      NewParamTs.push_back(Arg.getType());
      continue;
    }
    OptimizeableParams.push_back(PI);

    if (PI.isSret()) {
      continue; // optimizeable 'sret' parameter is removed in the clone
    }
    // parameter is converted from 'by pointer' to 'by value' passing, its type
    // is changed
    NewParamTs.push_back(PI.getOptimizedType());
  }
  if (OptimizeableParams.size() == 0) {
    return false; // nothing to optimize
  }
  // Optimize the function.
  Function *NewF = optimizeFunction(F, OptimizeableParams, NewParamTs);

  // Copy users to a separate container, to enable safe eraseFromParent
  // within optimizeCall.
  SmallVector<User *> FUsers;
  std::copy(F->users().begin(), F->users().end(), std::back_inserter(FUsers));

  // Optimize calls to the function.
  for (auto *U : FUsers) {
    auto *Call = cast<CallInst>(U);
    assert(Call->getCalledFunction() == F);
    optimizeCall(Call, NewF, OptimizeableParams);
  }
  NewF->takeName(F);
  return true;
}

namespace llvm {

PreservedAnalyses
ESIMDOptimizeVecArgCallConvPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool Modified = false;
  SmallPtrSet<Function *, 32> Visited;
#ifdef DEBUG_OPT_VEC_ARG_CALL_CONV
  {
    std::error_code EC;
    raw_fd_ostream Out("vec_arg_call_conv_in.ll", EC);
    M.print(Out, nullptr);
  }
#endif // DEBUG_OPT_VEC_ARG_CALL_CONV

  SmallVector<Function *, 16> ToErase;

  for (Function &F : M) {
    const bool FReplaced = processFunction(&F);
    Modified |= FReplaced;

    if (FReplaced) {
      ToErase.push_back(&F);
    }
  }
  std::for_each(ToErase.begin(), ToErase.end(),
                [](Function *F) { F->eraseFromParent(); });
#ifdef DEBUG_OPT_VEC_ARG_CALL_CONV
  {
    std::error_code EC;
    raw_fd_ostream Out("vec_arg_call_conv_out.ll", EC);
    M.print(Out, nullptr);
  }
#endif // DEBUG_OPT_VEC_ARG_CALL_CONV

  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
} // namespace llvm
