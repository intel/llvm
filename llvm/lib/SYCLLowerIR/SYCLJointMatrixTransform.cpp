//=== SYCLJointMatrixTransform.cpp - SYCL Joint Matrix transformation Pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which mutates Joint Matrix builtin calls to make them
// conformat with SPIR-V friendly LLVM IR specification.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLJointMatrixTransform.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {

static constexpr char ACCESS_CHAIN[] = "_Z19__spirv_AccessChain";
static constexpr char MATRIX_TYPE[] = "spirv.CooperativeMatrixKHR";
static constexpr char MATRIX_LAYOUT[] = "joint_matrix_layout_to_spv";

Type *getInnermostType(Type *Ty) {
  while (auto *ArrayTy = dyn_cast<ArrayType>(Ty))
    Ty = ArrayTy->getElementType();
  return Ty;
}

Type *replaceInnermostType(Type *Ty, Type *NewInnermostTy) {
  if (auto *ArrayTy = dyn_cast<ArrayType>(Ty))
    return ArrayType::get(
        replaceInnermostType(ArrayTy->getElementType(), NewInnermostTy),
        ArrayTy->getNumElements());
  return NewInnermostTy;
}

// This function is a copy of stripPointerCastsAndOffsets from Value.cpp,
// simplified and modified to strip non-zero GEP indices as well and also
// find nearest GEP instruction.
Value *stripPointerCastsAndOffsets(Value *V, bool StopOnGEP = false) {
  if (!V->getType()->isPointerTy())
    return V;

  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<Value *, 4> Visited;

  Visited.insert(V);
  do {
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      if (StopOnGEP && isa<GetElementPtrInst>(GEP))
        return V;
      V = GEP->getPointerOperand();
    } else if (auto *BC = dyn_cast<BitCastOperator>(V)) {
      Value *NewV = BC->getOperand(0);
      if (!NewV->getType()->isPointerTy())
        return V;
      V = NewV;
    } else if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V)) {
      V = ASC->getOperand(0);
    } else {
      if (auto *Call = dyn_cast<CallBase>(V)) {
        if (Value *RV = Call->getReturnedArgOperand()) {
          V = RV;
          // Strip the call instruction, since callee returns its RV
          // argument as return value. So, we need to continue stripping.
          continue;
        }
      }
      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);

  return V;
}

TargetExtType *extractMatrixType(StructType *WrapperMatrixTy) {
  if (!WrapperMatrixTy)
    return nullptr;
  TargetExtType *MatrixTy =
      dyn_cast<TargetExtType>(WrapperMatrixTy->getElementType(0));

  if (!MatrixTy)
    return nullptr;
  if (MatrixTy->getName() != MATRIX_TYPE)
    return nullptr;
  return MatrixTy;
}

// This function finds all calls to __spirv_AccessChain function and transforms
// its users and operands to make LLVM IR more SPIR-V friendly.
bool transformAccessChain(Function *F) {
  bool ModuleChanged = false;
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    User *U = *I++;
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    // This is a W/A for bfloat16 and tf32 types - they are represented in SYCL
    // as structures with int16/float storages. It means, that in LLVM IR
    // user of CallInst to __spirv_AccessChain function would be not load/store
    // instruction, but a zero GEP. This zero GEP is no-op, but can confuse a
    // SPIR-V consumer, so lets remove it here.
    auto *Unique = CI->getUniqueUndroppableUser();
    if (auto *GEP = dyn_cast_or_null<GetElementPtrInst>(Unique)) {
      if (GEP->hasAllZeroIndices()) {
        GEP->replaceAllUsesWith(CI);
        GEP->dropAllReferences();
        GEP->eraseFromParent();
      }
    }

    // It can happen that the optimizer can remove duplicated or dead uses
    // of CallInst to __spirv_AccessChain function. But it can't remove
    // __spirv_AccessChain call itself as it's a call to external function.
    // Lets clean such calls.
    if (CI->getNumUses() == 0) {
      CI->dropAllReferences();
      CI->eraseFromParent();
      continue;
    }

    // This routine extracts spirv.CooperativeMatrixKHR target extension type
    // from sycl::joint_matrix class object if it's used in __spirv_AccessChain
    // function call. It's necessary because otherwise OpAccessChain indices
    // would be wrong.
    Instruction *Ptr = dyn_cast<Instruction>(
        stripPointerCastsAndOffsets(CI->getArgOperand(0)));
    if (!Ptr || !isa<AllocaInst>(Ptr))
      continue;

    Type *AllocaTy = cast<AllocaInst>(Ptr)->getAllocatedType();
    // It may happen that sycl::joint_matrix class object is wrapped into
    // nested arrays. We need to find the innermost type to extract
    if (StructType *WrapperMatrixTy =
            dyn_cast<StructType>(getInnermostType(AllocaTy))) {
      TargetExtType *MatrixTy = extractMatrixType(WrapperMatrixTy);
      if (!MatrixTy)
        continue;

      AllocaInst *Alloca = nullptr;
      {
        IRBuilder Builder(CI);
        IRBuilderBase::InsertPointGuard IG(Builder);
        Builder.SetInsertPointPastAllocas(CI->getFunction());
        Alloca = Builder.CreateAlloca(replaceInnermostType(AllocaTy, MatrixTy));
        Alloca->takeName(Ptr);
      }
      Ptr->replaceAllUsesWith(Alloca);
      Ptr->dropAllReferences();
      Ptr->eraseFromParent();
      ModuleChanged = true;
    }

    // In case spirv.CooperativeMatrixKHR is used in arrays, we also need to
    // insert GEP to get pointer to target exention type and use it instead of
    // pointer to sycl::joint_matrix class object when it is passed to
    // __spirv_AccessChain
    // First we check if the argument came from a GEP instruction
    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(
        stripPointerCastsAndOffsets(CI->getArgOperand(0), /*StopOnGEP=*/true));
    if (!GEP)
      continue;

    // Check if GEP return type is a pointer to sycl::joint_matrix class object
    StructType *WrapperMatrixTy =
        dyn_cast<StructType>(GEP->getResultElementType());
    if (!extractMatrixType(WrapperMatrixTy))
      continue;

    // Insert GEP right before the __spirv_AccessChain call
    {
      IRBuilder Builder(CI);
      Value *NewGEP =
          Builder.CreateInBoundsGEP(WrapperMatrixTy, CI->getArgOperand(0),
                                    {Builder.getInt64(0), Builder.getInt32(0)});
      CI->setArgOperand(0, NewGEP);
      ModuleChanged = true;
    }
  }
  return ModuleChanged;
}

// Per SPIR-V specification Layout of a matrix must be a constant instruction
// aka a constexpr or specialization constant. Meanwhile in SYCL headers
// layout is passed as a parameter to joint_matrix_load function, so even if
// that layout is a constant expression in the user's code - it's not possible
// to prove that to the compiler, so constant propagation will happen only
// after inlining, not in AST. That means, that with O0 layout would remain
// to be a runtime variable in LLVM IR.
// SYCL matrix layout is being mapped on SPIR-V matrix layout by
// joint_matrix_layout_to_spv function. The following routine finds calls to
// this function and replaces them with the found constant.
// This function also cleans up code, that becomes dead. Pattern of the dead
// code is stable, as user's code doesn't affect it.
bool propagateConstexprLayout(Function *F) {
  llvm::SmallVector<Instruction *, 4> ToErase;
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    User *U = *I++;
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    auto *Op = dyn_cast<Instruction>(CI->getArgOperand(0));
    if (!Op || !isa<LoadInst>(Op))
      continue;
    auto *Ptr = dyn_cast<Instruction>(cast<LoadInst>(Op)->getPointerOperand());
    if (!Ptr)
      continue;

    ConstantInt *ConstLayout = nullptr;
    for (const auto &U : Ptr->users()) {
      if (!isa<StoreInst>(U))
        continue;
      assert(!ConstLayout && "More than 1 layout value was found");
      auto *SI = cast<StoreInst>(U);
      ConstLayout = dyn_cast<ConstantInt>(SI->getValueOperand());
      if (ConstLayout) {
        CI->replaceAllUsesWith(ConstLayout);
        ToErase.push_back(CI);
        ToErase.push_back(SI);
      }
    }
    if (ConstLayout) {
      ToErase.push_back(Op);
      ToErase.push_back(Ptr);
      if (auto *Cast = dyn_cast<AddrSpaceCastInst>(Ptr)) {
        auto *OrigPtr = Cast->getPointerOperand();
        if (auto *AI = dyn_cast<AllocaInst>(OrigPtr)) {
          ToErase.push_back(AI);
        }
      }
    }
  }
  for (Instruction *II : ToErase) {
    if (!II)
      continue;
    II->dropAllReferences();
    II->eraseFromParent();
  }
  return !ToErase.empty();
}
} // namespace

PreservedAnalyses
SYCLJointMatrixTransformPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  llvm::SmallVector<Function *, 1> ToErase;
  for (Function &F : M) {
    if (!F.isDeclaration()) {
      if (F.getName() == MATRIX_LAYOUT) {
        ModuleChanged |= propagateConstexprLayout(&F);
        ToErase.push_back(&F);
      } else
        continue;
    }
    if (F.getName().starts_with(ACCESS_CHAIN))
      ModuleChanged |= transformAccessChain(&F);
  }

  for (auto *F : ToErase)
    if (F->users().empty())
      F->eraseFromParent();

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
