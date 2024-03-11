//===- SPIRVTypeScavenger.cpp - Recover pointer types in opaque pointer IR ===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2022 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of The Khronos Group, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements the necessary logic to recover pointer types from LLVM
// IR for the output SPIR-V file from the opaque pointers in LLVM IR.
//
// The core algorithm being implemented is rather simple, although there are
// several complications that make its implementation more difficult. At its
// core, the algorithm works like this:
//
// 1. Replace every instance of an opaque pointer type with a typed pointer that
//    points to an unknown type variable.
// 2. Convert each instruction into a series of typing rules. For example,
//    load i8, ptr %ptr implies that %ptr must be typedptr(i8, 0).
// 3. Based on the typing rules, resolve the type variables to concrete types.
// 4. If the typing rules produce a contradiction (e.g., i8 == i32), insert a
//    synthetic bitcast to represent the bitcast that would have been present in
//    a typed pointer IR.
// 5. If any type variables are unresolved at the end of the typing process,
//    assign i8 to them instead.
//
// Typed pointers are represented with the TypedPointerType. Type variables are
// represented as target("typevar", i), where i is an integer to disambiguate
// between different unknown types. (It is an index into the TypeVariables and
// UnifiedTypeVars fields).
//
// Step 3 of the above algorithm is represented by unifyType, which implements a
// unification-based type algorithm. This means there exists essentially just
// four cases that need to be considered:
// * unify(type var, concrete type):
//   In this case, the concrete type is assigned to the type variable
//   Note: It is possible for concrete type to contain nested type variables,
//         e.g., typedptr(target("typevar", 3), 4)
// * unify(type var, concrete type containing type var):
//   Unification fails in this case. This can come up if you have code like
//   this:
//     %ptr = alloca ptr
//     store ptr %ptr, ptr %ptr
// * unify(type var, type var):
//   In this case, the two type variables are unified into one so that they get
//   the same concrete type. This uses IntEqClasses as the implementation of the
//   union-find data structure.
// * unify(concrete type, same concrete type):
//   There is nothing to do in this case
// * unify(concrete type, different concrete type):
//   Unification fails in this case, and a bitcast needs to be generated.
//
// Note that this algorithm does not attempt to seek a minimal set of bitcasts
// that need to be added to produce a correctly-typed program.
//
// Type rules are represented by the SPIRVTypeScavenger::TypeRule class, and
// should be constructed using the provided static methods (which are easier to
// understand than the constructor itself). Type rules boil down to the
// following categories:
// * operand I has type T
// * operand I has the same type as operand J
// * the return value has type T
// * the return value has the same type as operand I
// with any of the above operands, types, or return values potentially having a
// level of indirection. For example, the rule for an addrspacecast is that the
// return value points to the same type that its sole operand points to. The
// indirection effectively means T->getScalarType()->getPointerElementType().
// Note: When constructing type rules fixing an operand or the return to a
//       particular type T, the type must be a type using typed pointers and/or
//       type variables in lieu of ptr.
// Note: getTypeRules may be called twice on an instruction, so if a new type
//       variable needs to be created for type rules, it needs to be saved in
//       the AssociatedTypeVariables method to ensure proper functioning. This
//       is particularly important if you need to use the type variable both in
//       constraining the return value and an operand.
//
// Now for the complications to the above algorithm:
//
// The most notable issue is that LLVM does not allow no-op constant
// expressions to be created. This means that we have to be very careful about
// the types of constant values. As the SPIR-V translator expands most constant
// expressions into instructions, this isn't much of an issue, but where it
// really comes into play is with global variable initializers, which don't have
// that luxury. Global variable typing therefore pays careful attention to the
// type of the initializer.
//
// Constructing type variables is a slightly expensive step, so before
// attempting to create a type variable for the return of an instruction, we
// instead look through the type rules to see if we can get the type of the
// return value from one of its input operands.
//
// Recursive pointer types are not possible to express, as we make no attempt
// to recover pointer types within struct types. It is still possible for
// a type rule to suggest an infinite recursive type (consider the example
// store ptr %x, ptr %x), so we have to guard against it in unifyType.
//
//===----------------------------------------------------------------------===//

#include "SPIRVTypeScavenger.h"

#include "SPIRVInternal.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Regex.h"

#define DEBUG_TYPE "type-scavenger"

using namespace llvm;

namespace {
static inline std::optional<unsigned> isTypeVariable(Type *T) {
  if (auto *TET = dyn_cast<TargetExtType>(T))
    if (TET->getName() == "typevar") {
      return TET->getIntParameter(0);
    }
  return std::nullopt;
}

/// Convert Ty to a type that can be unified with a type-variable-ified L, given
/// that either or both types may have an indirection.
/// For example, adjust(L, true, Ty, false) will extract the element type of Ty
/// for unifying with L.
/// In this method, L is expected to be a type of a value, while Ty (and the
/// return value) will use TypedPointerType instead of PointerType.
static Type *adjustIndirect(Type *L, bool LIndirect, Type *Ty, bool TIndirect) {
  if (LIndirect)
    Ty = cast<TypedPointerType>(Ty->getScalarType())->getElementType();
  if (TIndirect) {
    unsigned AS = L->getScalarType()->getPointerAddressSpace();
    Ty = TypedPointerType::get(Ty, AS);
    if (auto *VT = dyn_cast<VectorType>(L))
      Ty = VectorType::get(Ty, VT->getElementCount());
  }
  return Ty;
}

/// Return the type with all inner pointer types replaced with the result of
/// calling MutatePointer(PointerAddressSpace).
template <typename Fn> Type *mutateType(Type *T, Fn MutatePointer) {
  if (T->isPointerTy()) {
    return MutatePointer(T->getPointerAddressSpace());
  }
  if (auto *VT = dyn_cast<VectorType>(T)) {
    return VectorType::get(mutateType(VT->getScalarType(), MutatePointer),
                           VT->getElementCount());
  }
  if (auto *AT = dyn_cast<ArrayType>(T)) {
    return ArrayType::get(mutateType(AT->getElementType(), MutatePointer),
                          AT->getNumElements());
  }
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    SmallVector<Type *, 4> ParamTypes;
    for (Type *Inner : FT->params())
      ParamTypes.push_back(mutateType(Inner, MutatePointer));
    Type *ReturnTy = mutateType(FT->getReturnType(), MutatePointer);
    return FunctionType::get(ReturnTy, ParamTypes, FT->isVarArg());
  }
  // TODO: support literal structs
  return T;
}

/// Return true if the type is an opaque pointer, or contains an opaque pointer
/// that needs to be typed.
bool hasPointerType(Type *T) {
  if (T->isPtrOrPtrVectorTy())
    return true;
  if (auto *AT = dyn_cast<ArrayType>(T))
    return hasPointerType(AT->getElementType());
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    for (Type *Inner : FT->params())
      if (hasPointerType(Inner))
        return true;
    return hasPointerType(FT->getReturnType());
  }
  // TODO: literal structs
  return false;
}

/// Get a type where all internal pointer types are replaced with i8*.
Type *getUnknownTyped(Type *T) {
  Type *Int8Ty = Type::getInt8Ty(T->getContext());
  return mutateType(
      T, [=](unsigned AS) { return TypedPointerType::get(Int8Ty, AS); });
}

bool hasTypeVariable(Type *T, const unsigned TypeVarNum) {
  if (auto *TPT = dyn_cast<TypedPointerType>(T))
    return hasTypeVariable(TPT->getElementType(), TypeVarNum);
  if (auto *VT = dyn_cast<VectorType>(T))
    return hasTypeVariable(VT->getElementType(), TypeVarNum);
  if (auto *AT = dyn_cast<ArrayType>(T))
    return hasTypeVariable(AT->getElementType(), TypeVarNum);
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    for (Type *Inner : FT->params())
      if (hasTypeVariable(Inner, TypeVarNum))
        return true;
    return hasTypeVariable(FT->getReturnType(), TypeVarNum);
  }
  if (auto CheckNum = isTypeVariable(T)) {
    return TypeVarNum == *CheckNum;
  }
  return false;
}
} // anonymous namespace

Type *SPIRVTypeScavenger::substituteTypeVariables(Type *T) {
  if (auto *TPT = dyn_cast<TypedPointerType>(T))
    return TypedPointerType::get(substituteTypeVariables(TPT->getElementType()),
                                 TPT->getAddressSpace());
  if (auto *VT = dyn_cast<VectorType>(T))
    return VectorType::get(substituteTypeVariables(VT->getElementType()),
                           VT->getElementCount());
  if (auto *AT = dyn_cast<ArrayType>(T))
    return ArrayType::get(substituteTypeVariables(AT->getElementType()),
                          AT->getNumElements());
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    SmallVector<Type *, 4> ParamTypes;
    for (Type *Inner : FT->params())
      ParamTypes.push_back(substituteTypeVariables(Inner));
    Type *ReturnTy = substituteTypeVariables(FT->getReturnType());
    return FunctionType::get(ReturnTy, ParamTypes, FT->isVarArg());
  }
  if (auto Index = isTypeVariable(T)) {
    unsigned TypeVarNum = *Index;
    TypeVarNum = UnifiedTypeVars.findLeader(TypeVarNum);
    Type *&SubstTy = TypeVariables[TypeVarNum];
    // A value in TypeVariables may itself contain type variables that need to
    // be substituted. Substitute these as well.
    if (SubstTy)
      return SubstTy = substituteTypeVariables(SubstTy);
    // Even if it's not fully resolved, return the leader of the current
    // equivalence class instead. This allows for easier scanning of recursive
    // type declarations.
    return TargetExtType::get(T->getContext(), "typevar", {}, {TypeVarNum});
  }
  return T;
}

bool SPIRVTypeScavenger::unifyType(Type *T1, Type *T2) {
  T1 = substituteTypeVariables(T1);
  T2 = substituteTypeVariables(T2);
  if (T1 == T2)
    return true;

  auto SetTypeVar = [&](unsigned TypeVarNum, Type *ActualTy) {
    unsigned Leader = UnifiedTypeVars.findLeader(TypeVarNum);

    // This method might be called with T1 as a concrete type containing
    // pointers, and we want to make sure those don't leak into type variables.
    // Guard against that here.
    ActualTy = allocateTypeVariable(ActualTy);

    // Check for recursion in type variables. Such recursive types generally
    // cannot be correctly typed.
    if (hasTypeVariable(ActualTy, Leader))
      return false;

    LLVM_DEBUG(dbgs() << "Type variable " << TypeVarNum << " is " << *ActualTy
                      << "\n");
    assert(!TypeVariables[Leader] && "Type was already fixed?");
    TypeVariables[Leader] = ActualTy;
    return true;
  };

  if (auto T1Num = isTypeVariable(T1)) {
    if (auto T2Num = isTypeVariable(T2)) {
      // Two type variables. Unify the two of them into the same type.
      if (T1Num != T2Num) {
        UnifiedTypeVars.join(*T1Num, *T2Num);
        LLVM_DEBUG(dbgs() << "Joining typevar " << *T1Num << " and " << *T2Num
                          << "\n");
      }
      return true;
    }
    return SetTypeVar(*T1Num, T2);
  }

  if (auto T2Num = isTypeVariable(T2)) {
    // We know that T1 can't be a type variable, so the only possibility is that
    // we assign T2 to T1.
    return SetTypeVar(*T2Num, T1);
  }

  // At this point, we know that neither type is a type variable. If the two
  // types have a different structure, we can't unify them.
  if (auto *TPT1 = dyn_cast<TypedPointerType>(T1)) {
    if (auto *TPT2 = dyn_cast<TypedPointerType>(T2)) {
      if (TPT1->getAddressSpace() != TPT2->getAddressSpace())
        return false;
      return unifyType(TPT1->getElementType(), TPT2->getElementType());
    }
    return false;
  }

  // We can also call unifyType(ptr, T2) (this is useful for propagating types
  // to return values of instructions). In such a case, the ptr type is
  // equivalent to typedptr(target("typevar")), for some type variable we
  // haven't yet allocated. In this use case, it suffices to know that T2 is
  // also a typed pointer type, as the case where T2 is a type variable was
  // handled earlier.
  if (isa<PointerType>(T1)) {
    if (auto *TPT2 = dyn_cast<TypedPointerType>(T2))
      return TPT2->getAddressSpace() == T1->getPointerAddressSpace();
    return false;
  }

  if (auto *FT1 = dyn_cast<FunctionType>(T1)) {
    if (auto *FT2 = dyn_cast<FunctionType>(T2)) {
      if (FT1->getNumParams() != FT2->getNumParams())
        return false;
      if (FT1->isVarArg() != FT2->isVarArg())
        return false;
      if (!unifyType(FT1->getReturnType(), FT2->getReturnType()))
        return false;
      for (const auto &[PT1, PT2] : zip(FT1->params(), FT2->params()))
        if (!unifyType(PT1, PT2))
          return false;
      return true;
    }
    return false;
  }

  if (auto *VT1 = dyn_cast<VectorType>(T1)) {
    if (auto *VT2 = dyn_cast<VectorType>(T2)) {
      if (VT1->getElementCount() != VT2->getElementCount())
        return false;
      return unifyType(VT1->getScalarType(), VT2->getScalarType());
    }
    return false;
  }

  if (auto *AT1 = dyn_cast<ArrayType>(T1)) {
    if (auto *AT2 = dyn_cast<ArrayType>(T2)) {
      if (AT1->getNumElements() != AT2->getNumElements())
        return false;
      return unifyType(AT1->getElementType(), AT2->getElementType());
    }
    return false;
  }

  // We already established T1 != T2 earlier, so there's no way we're capable of
  // unifying at this point.
  return false;
}

void SPIRVTypeScavenger::typeModule(Module &M) {
  // Generate corrected function types for all functions in the module.
  for (auto &F : M.functions()) {
    deduceFunctionType(F);
  }

  // Now that we have function types, type the global variables. We have
  // restrictions on our ability to do typing on constant initializers, so we
  // need to make sure that global variables get typed.
  for (auto &GV : M.globals())
    typeGlobalValue(GV, GV.hasInitializer() ? GV.getInitializer() : nullptr);

  // SPIR-V doesn't support global aliases, so pass through all of the types of
  // global aliasees to the global alias (this at least ensures correct typing
  // of uses of the global alias).
  for (auto &GA : M.aliases()) {
    Type *ScavengedTy = getScavengedType(GA.getAliasee());
    DeducedTypes[&GA] = ScavengedTy;
    LLVM_DEBUG(dbgs() << "Type of " << GA << " is " << *ScavengedTy << "\n");
  }

  // Type all instructions in the module.
  for (auto &F : M.functions()) {
    LLVM_DEBUG(dbgs() << "Typing function " << F.getName() << "\n");
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        getTypeAfterRules(&I);
        correctUseTypes(I);
      }
    }
  }

  // If there are any type variables we couldn't resolve, fallback to assigning
  // them as an i8* type.
  Type *Int8Ty = Type::getInt8Ty(M.getContext());
  for (const auto &[TypeVarNum, TypeVar] : enumerate(TypeVariables)) {
    unsigned PrimaryVar = UnifiedTypeVars.findLeader(TypeVarNum);
    Type *LeaderTy = TypeVariables[PrimaryVar];
    if (TypeVar)
      TypeVar = substituteTypeVariables(TypeVar);
    if (LeaderTy)
      LeaderTy = substituteTypeVariables(LeaderTy);
    assert((!TypeVar || LeaderTy == TypeVar) &&
           "Inconsistent type variable unification");
    if (!TypeVar) {
      TypeVar = LeaderTy ? LeaderTy : Int8Ty;
    }
    TypeVariables[TypeVarNum] = TypeVar;
    LLVM_DEBUG(dbgs() << "Type variable " << TypeVarNum << " resolved to "
                      << *TypeVar << "\n");
  }
  return;
}

bool SPIRVTypeScavenger::typeIntrinsicCall(
    CallBase &CB, SmallVectorImpl<TypeRule> &TypeRules) {
  Function *TargetFn = CB.getCalledFunction();
  assert(TargetFn && TargetFn->isDeclaration() &&
         "Call is not an intrinsic function call");
  LLVMContext &Ctx = TargetFn->getContext();

  // If the type is a pointer type, replace it with a typedptr(typevar) type
  // instead, using AssociatedTypeVariables.
  auto GetTypeOrTypeVar = [&](Type *BaseTy) {
    if (!BaseTy->isPointerTy())
      return BaseTy;
    Type *&AssociatedTy = AssociatedTypeVariables[&CB];
    if (!AssociatedTy)
      AssociatedTy = allocateTypeVariable(BaseTy);
    return AssociatedTy;
  };

  StringRef DemangledName;
  if (oclIsBuiltin(TargetFn->getName(), DemangledName) ||
      isDecoratedSPIRVFunc(TargetFn, DemangledName)) {
    Op OC = getSPIRVFuncOC(DemangledName);
    switch (OC) {
    case OpAtomicLoad:
    case OpAtomicExchange:
    case OpAtomicCompareExchange:
    case OpAtomicIAdd:
    case OpAtomicISub:
    case OpAtomicFAddEXT:
    case OpAtomicSMin:
    case OpAtomicUMin:
    case OpAtomicFMinEXT:
    case OpAtomicSMax:
    case OpAtomicUMax:
    case OpAtomicFMaxEXT:
    case OpAtomicAnd:
    case OpAtomicOr:
    case OpAtomicXor:
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, CB.getType()));
      return true;
    case OpAtomicStore:
      TypeRules.push_back(
          TypeRule::pointsTo(CB, 0, CB.getArgOperand(3)->getType()));
      return true;
    case OpGenericCastToPtr:
    case OpGenericCastToPtrExplicit: {
      Type *Ty =
          cast<TypedPointerType>(getFunctionType(TargetFn)->getParamType(0))
              ->getElementType();
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Ty));
      TypeRules.push_back(TypeRule::returnsPointerTo(Ty));
      return true;
    }
    default:
      // Do nothing
      break;
    }
  }

  if (auto IntrinID = TargetFn->getIntrinsicID()) {
    switch (IntrinID) {
    case Intrinsic::memcpy: {
      // First two parameters are pointers, but they point to the same thing
      // (albeit maybe in different address spaces).
      TypeRules.push_back(TypeRule::isIndirect(CB, 0, 1));
      break;
    }
    case Intrinsic::memset:
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::invariant_start:
      // These intrinsics were stored as i8* as typed pointers, and the SPIR-V
      // writer will expect these to be i8*, even if they can be any pointer
      // type.
      TypeRules.push_back(TypeRule::pointsTo(CB, 1, Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::invariant_end:
      // This is like invariant_start with an extra string parameter in the
      // beginning (so the pointer object moves to argument two).
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Type::getInt8Ty(Ctx)));
      TypeRules.push_back(TypeRule::pointsTo(CB, 2, Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::var_annotation:
      // The first parameter of these is an i8*.
      // (See below for notes on the latter parameters).
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::ptr_annotation:
      // Returns the first argument.
      // (See below for notes on the latter parameters).
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Type::getInt8Ty(Ctx)));
      TypeRules.push_back(TypeRule::returnsPointerTo(Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::annotation:
      // Second and third parameters are strings, which should be constants
      // for global variables. Nominally, this is i8*, but we specifically
      // *do not* want to insert bitcast instructions (they need to remain
      // global constants).
      break;
    case Intrinsic::stacksave:
      TypeRules.push_back(TypeRule::returnsPointerTo(Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::stackrestore:
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::instrprof_cover:
    case Intrinsic::instrprof_increment:
    case Intrinsic::instrprof_increment_step:
    case Intrinsic::instrprof_value_profile:
      // llvm.instrprof.* intrinsics are not supported
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, Type::getInt8Ty(Ctx)));
      break;
    case Intrinsic::masked_gather: {
      Type *ScalarTy = GetTypeOrTypeVar(CB.getType()->getScalarType());
      TypeRules.push_back(TypeRule::pointsTo(CB, 0, ScalarTy));
      if (CB.getType()->getScalarType()->isPointerTy())
        TypeRules.push_back(TypeRule::propagates(CB, 3));
      break;
    }
    case Intrinsic::masked_scatter: {
      Type *ScalarTy =
          GetTypeOrTypeVar(CB.getOperand(0)->getType()->getScalarType());
      TypeRules.push_back(TypeRule::pointsTo(CB, 1, ScalarTy));
      break;
    }
    default:
      return false;
    }
  } else if (TargetFn->getName().starts_with("_Z18__spirv_ocl_printf")) {
    Type *Int8Ty = Type::getInt8Ty(Ctx);
    // The first argument is a string pointer. Subsequent arguments may include
    // pointer-valued arguments, corresponding to %s or %p parameters.
    // Therefore, all parameters need to be i8*.
    for (Use &U : CB.args()) {
      if (U->getType()->isPointerTy())
        TypeRules.push_back(TypeRule::pointsTo(U, Int8Ty));
    }
  } else if (TargetFn->getName() == "__spirv_GetKernelWorkGroupSize__") {
    TypeRules.push_back(TypeRule::pointsTo(CB, 1, Type::getInt8Ty(Ctx)));
  } else if (TargetFn->getName() ==
             "__spirv_GetKernelPreferredWorkGroupSizeMultiple__") {
    TypeRules.push_back(TypeRule::pointsTo(CB, 1, Type::getInt8Ty(Ctx)));
  } else if (TargetFn->getName() ==
             "__spirv_GetKernelNDrangeMaxSubGroupSize__") {
    TypeRules.push_back(TypeRule::pointsTo(CB, 2, Type::getInt8Ty(Ctx)));
  } else if (TargetFn->getName() == "__spirv_GetKernelNDrangeSubGroupCount__") {
    TypeRules.push_back(TypeRule::pointsTo(CB, 2, Type::getInt8Ty(Ctx)));
  } else if (TargetFn->getName().starts_with("__spirv_EnqueueKernel__")) {
    Type *DevEvent = TargetExtType::get(Ctx, "spirv.DeviceEvent");
    TypeRules.push_back(TypeRule::pointsTo(CB, 4, DevEvent));
    TypeRules.push_back(TypeRule::pointsTo(CB, 5, DevEvent));
    TypeRules.push_back(TypeRule::pointsTo(CB, 7, Type::getInt8Ty(Ctx)));
  } else if (TargetFn->getName().starts_with(
                 "_Z33__regcall3____builtin_invoke_simd")) {
    // First argument is a function to call, subsequent arguments are parameters
    // to said function.
    auto *FnTy = getFunctionType(cast<Function>(CB.getArgOperand(0)));
    TypeRules.push_back(TypeRule::pointsTo(CB, 0, FnTy));
    typeFunctionParams(CB, FnTy, 1, true, TypeRules);
    // Also apply type rules to the parameter types of the underlying function.
    return false;
  } else
    return false;

  return true;
}

void SPIRVTypeScavenger::typeFunctionParams(
    CallBase &CB, FunctionType *FT, unsigned ArgStart, bool IncludeRet,
    SmallVectorImpl<TypeRule> &TypeRules) {
  for (const auto &[U, ArgTy] :
       zip(drop_begin(CB.args(), ArgStart), FT->params())) {
    if (hasPointerType(U->getType())) {
      TypeRules.push_back(TypeRule::is(U, ArgTy));
    }
  }
  if (IncludeRet) {
    if (hasPointerType(CB.getType()))
      TypeRules.push_back(TypeRule::returns(FT->getReturnType()));
  }
}

void SPIRVTypeScavenger::typeGlobalValue(GlobalValue &GV, Constant *Init) {
  auto GetNaturalType = [&](Value *C) -> Type * {
    if (isa<GlobalValue>(C)) {
      auto It = DeducedTypes.find(C);
      if (It != DeducedTypes.end())
        return It->second;
    }

    return getUnknownTyped(C->getType());
  };

  Type *Ty = GV.getValueType();
  Type *MemType = nullptr;
  // If the initializer is an array or vector of globals that all have the same
  // type, prefer to use that type.
  if (Init && (isa<ConstantArray>(Init) || isa<ConstantVector>(Init))) {
    Type *InnerTy = Init->getType()->getContainedType(0);
    if (InnerTy->isPointerTy()) {
      Type *CommonTy = allocateTypeVariable(InnerTy);
      bool Successful = true;
      for (Value *Op : Init->operand_values()) {
        Successful &= unifyType(CommonTy, GetNaturalType(Op));
        if (!Successful)
          break;
      }
      if (Successful) {
        CommonTy = substituteTypeVariables(CommonTy);
        if (isa<ConstantArray>(Init))
          MemType = ArrayType::get(CommonTy, Ty->getArrayNumElements());
        else
          MemType = VectorType::get(CommonTy,
                                    cast<VectorType>(Ty)->getElementCount());
      }
    }
  }

  // If there's an initializer, give it a fixed type based on the initializer.
  if (Init && !MemType)
    MemType = GetNaturalType(Init);

  // At this point, use a fixed type based on the value type of the global value
  // if we didn't compute it already.
  if (!MemType)
    MemType = getUnknownTyped(GV.getValueType());

  Type *TypedTy = TypedPointerType::get(MemType, GV.getAddressSpace());
  LLVM_DEBUG(dbgs() << "@" << GV.getName() << " has type " << *TypedTy << "\n");
  DeducedTypes[&GV] = TypedTy;
}

static Type *getParamType(const AttributeList &AL, unsigned ArgNo) {
  if (Type *Ty = AL.getParamByValType(ArgNo))
    return Ty;
  if (Type *Ty = AL.getParamStructRetType(ArgNo))
    return Ty;
  if (Type *Ty = AL.getParamElementType(ArgNo))
    return Ty;
  if (Type *Ty = AL.getParamInAllocaType(ArgNo))
    return Ty;
  if (Type *Ty = AL.getParamPreallocatedType(ArgNo))
    return Ty;
  return nullptr;
}

void SPIRVTypeScavenger::deduceFunctionType(Function &F) {
  // Start by constructing a basic function type that replaces all pointer
  // types in arguments (and the return type) with type variables. We may
  // resolve those type variables almost immediately, but this is a starting
  // point.
  FunctionType *FuncTy = F.getFunctionType();
  if (hasPointerType(FuncTy))
    FuncTy = cast<FunctionType>(allocateTypeVariable(F.getFunctionType()));
  DeducedTypes[&F] = TypedPointerType::get(FuncTy, F.getAddressSpace());

  auto TypeArgument = [&](Argument *Arg, Type *T) {
    [[maybe_unused]] bool Successful =
        unifyType(FuncTy->getParamType(Arg->getArgNo()), T);
    assert(Successful && "Unification of argument type failed?");
    LLVM_DEBUG(dbgs() << "  Arg " << *Arg << " is known to be " << *T << "\n");
    DeducedTypes[Arg] = T;
  };

  // Gather a list of arguments that have unresolved type variables.
  SmallVector<Argument *, 8> PointerArgs;
  for (Argument &Arg : F.args()) {
    DeducedTypes[&Arg] = FuncTy->getParamType(Arg.getArgNo());
    if (hasPointerType(Arg.getType()))
      PointerArgs.push_back(&Arg);
  }

  // Get any arguments from attributes where possible.
  for (Argument *Arg : PointerArgs) {
    Type *Ty = getParamType(F.getAttributes(), Arg->getArgNo());
    if (Ty)
      TypeArgument(Arg, TypedPointerType::get(
                            Ty, Arg->getType()->getPointerAddressSpace()));
  }

  // The first non-sret argument of block_invoke functions is the block capture
  // struct, which should be passed as an i8*.
  static const Regex BlockInvokeRegex(
      "^(__.+)?_block_invoke(_[0-9]+)?(_kernel)?$");
  if (BlockInvokeRegex.match(F.getName())) {
    for (Argument *Arg : PointerArgs) {
      if (!Arg->hasAttribute(Attribute::StructRet)) {
        TypeArgument(Arg, getUnknownTyped(Arg->getType()));
        break;
      }
    }
  }

  // If the function is a mangled name, try to recover types from the Itanium
  // name mangling. Do this only for function types that without bodies, where
  // existing code can propagate types to the parameters.
  // TODO: Investigate if target extension types and the specially-handled
  // SPIR-V intrinsics renders this code unnecessary.
  if (F.isDeclaration() && F.getName().starts_with("_Z")) {
    if (F.getName().starts_with("_Z")) {
      SmallVector<Type *, 8> ParamTypes;
      if (getParameterTypes(&F, ParamTypes)) {
        for (Argument *Arg : PointerArgs) {
          if (auto *Ty =
                  dyn_cast<TypedPointerType>(ParamTypes[Arg->getArgNo()]))
            if (!Arg->hasAttribute(Attribute::StructRet))
              TypeArgument(Arg, Ty);
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Type of @" << F.getName() << " is "
                    << *substituteTypeVariables(FuncTy) << "\n");
}

/// Certain constant types (null, undef, and poison) will get their type from
/// the use of the constant. We discover the type of the use by inserting a
/// synthetic bitcast instruction before the use. For these types, we need to
/// have special handling in a few places, and this indicates that it needs to
/// be done.
static bool doesNotImplyType(Value *V) {
  return isa<ConstantPointerNull>(V) || isa<UndefValue>(V);
}

Type *SPIRVTypeScavenger::getTypeAfterRules(Value *V) {
  auto *Ty = V->getType();
  if (!hasPointerType(Ty))
    return Ty;

  // Don't try to store null, undef, or poison in our type map. We'll call these
  // i8* by default; if any use has a different type, a bitcast will be added
  // later.
  if (doesNotImplyType(V)) {
    return getUnknownTyped(Ty);
  }

  // Check if we've already deduced a type for the value.
  Type *KnownType = DeducedTypes.lookup(V);
  if (KnownType)
    return substituteTypeVariables(KnownType);

  assert(
      !isa<GlobalValue>(V) && !isa<Argument>(V) &&
      "Globals and arguments must be fully handled before calling this method");

  // All constants will have their pointer types handled as i8*.
  if (!isa<Instruction>(V))
    return getUnknownTyped(Ty);

  assert(!is_contained(VisitStack, V) && "Found cycle in type scavenger");
  VisitStack.push_back(V);

  // Try to propagate from type rules constraining the return value.
  SmallVector<TypeRule, 4> TypeRules;
  getTypeRules(*cast<Instruction>(V), TypeRules);
  for (TypeRule &Rule : TypeRules) {
    if (Rule.OpNo != RETURN_OPERAND)
      continue;

    // Get the target type from the rule. If it comes from an operand,
    // recursively attempt to find the type from the operand (but avoid any
    // cycles).
    Type *TargetTy;
    if (auto *UsedTy = dyn_cast<Type *>(Rule.Target)) {
      TargetTy = allocateTypeVariable(UsedTy);
    } else {
      Value *Arg = cast<Use *>(Rule.Target)->get();
      if (is_contained(VisitStack, Arg))
        continue;

      Value *Source = cast<Use *>(Rule.Target)->get();
      // If the source argument is null, undef, or poison, then move on to
      // another rule to give better type hints.
      if (doesNotImplyType(Source))
        continue;

      TargetTy = substituteTypeVariables(getTypeAfterRules(Source));
    }

    // If the argument is a null pointer, try another operand instead.
    if (!TargetTy)
      continue;
    KnownType =
        adjustIndirect(Ty, Rule.LhsIndirect, TargetTy, Rule.RhsIndirect);
    // Make sure that the type is consistent with the type format of Ty.
    if (!unifyType(Ty, KnownType))
      KnownType = nullptr;
    break;
  }

  // If we still haven't gotten a type at this point, just construct a new type
  // variable and rely on later uses to recover the type.
  if (!KnownType) {
    LLVM_DEBUG(dbgs() << *V << " matched no typing rules\n");
    KnownType = allocateTypeVariable(Ty);
  }

  DeducedTypes[V] = KnownType;
  VisitStack.pop_back();

  LLVM_DEBUG(dbgs() << "Assigned type " << *KnownType << " to " << *V << "\n");
  return KnownType;
}

void SPIRVTypeScavenger::getTypeRules(Instruction &I,
                                      SmallVectorImpl<TypeRule> &TypeRules) {
  auto GetAssociatedTypeVariable = [&](Type *T) {
    Type *&TypeVar = AssociatedTypeVariables[&I];
    if (TypeVar)
      return TypeVar;
    return TypeVar = allocateTypeVariable(T);
  };

  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    Type *GepTy = GEP->getSourceElementType();
    Type *ReturnTy = GEP->getResultElementType();
    if (hasPointerType(GepTy)) {
      GepTy = GetAssociatedTypeVariable(GepTy);
      // Iterate the indices to find the return type, based on the version of
      // the type using type variables and typed pointer types instead.
      ReturnTy = GepTy;
      for (Use &U : drop_begin(GEP->indices()))
        ReturnTy = GetElementPtrInst::getTypeAtIndex(ReturnTy, U.get());
    } else {
      // It's possible that ReturnTy might be a type containing a ptr. However,
      // if we aren't typing the struct type specifically, then this type is
      // going to be coerced by the writer to i8*, so don't allocate any type
      // variables for it.
      ReturnTy = getUnknownTyped(ReturnTy);
    }
    TypeRules.push_back(TypeRule::pointsTo(I, 0, GepTy));
    TypeRules.push_back(TypeRule::returnsPointerTo(ReturnTy));
  } else if (isa<LoadInst>(&I)) {
    TypeRules.push_back(
        TypeRule::pointsToReturn(I, LoadInst::getPointerOperandIndex()));
  } else if (isa<StoreInst>(&I)) {
    TypeRules.push_back(
        TypeRule::pointsTo(I, StoreInst::getPointerOperandIndex(), 0U));
  } else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(&I)) {
    TypeRules.push_back(
        TypeRule::pointsTo(I, AtomicCmpXchgInst::getPointerOperandIndex(), 1));
    if (hasPointerType(AI->getCompareOperand()->getType()))
      TypeRules.push_back(TypeRule::is(I, 1, 2));
  } else if (auto *AI = dyn_cast<AtomicRMWInst>(&I)) {
    TypeRules.push_back(
        TypeRule::pointsTo(I, AtomicRMWInst::getPointerOperandIndex(), 1));
    if (hasPointerType(AI->getValOperand()->getType()))
      TypeRules.push_back(TypeRule::propagates(I, 1));
  } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
    TypeRules.push_back(TypeRule::returnsPointerTo(AI->getAllocatedType()));
  } else if (auto *CI = dyn_cast<ICmpInst>(&I)) {
    // icmp can compare pointers. If it isn't, ignore the instruction.
    if (!hasPointerType(CI->getOperand(0)->getType()))
      return;

    // The two pointer operands should have the same type.
    TypeRules.push_back(TypeRule::is(I, 1, 0));
  } else if (auto *SI = dyn_cast<SelectInst>(&I)) {
    if (!hasPointerType(SI->getType()))
      return;

    // Both selected values should have the same type as the result.
    TypeRules.push_back(TypeRule::propagates(I, 1));
    TypeRules.push_back(TypeRule::propagates(I, 2));
  } else if (auto *Phi = dyn_cast<PHINode>(&I)) {
    if (!hasPointerType(Phi->getType()))
      return;

    for (Use &U : Phi->incoming_values()) {
      TypeRules.push_back(TypeRule::propagates(U));
    }
  } else if (isa<FreezeInst>(&I)) {
    if (!hasPointerType(I.getType()))
      return;
    TypeRules.push_back(TypeRule::propagates(I, 0));
  } else if (auto *AS = dyn_cast<AddrSpaceCastInst>(&I)) {
    TypeRules.push_back(TypeRule::propagatesIndirect(*AS, 0));
  } else if (isa<ReturnInst>(&I)) {
    if (!hasPointerType(I.getFunction()->getReturnType()))
      return;
    Type *ExpectedTy = getFunctionType(I.getFunction())->getReturnType();
    TypeRules.push_back(TypeRule::is(0, ExpectedTy));
  } else if (auto *CB = dyn_cast<CallBase>(&I)) {
    // If we have an identified function for the call instruction, map the
    // arguments we pass in to the argument requirements of the function.
    if (Function *F = CB->getCalledFunction()) {
      if (!F->isDeclaration() || !typeIntrinsicCall(*CB, TypeRules)) {
        typeFunctionParams(*CB, getFunctionType(F), 0, true, TypeRules);
      }
    } else {
      // In the case of function pointers, we need to also assert the function
      // type of the call instruction itself.
      // In the case of inline assembly, the inline asm type is typed as if all
      // ptr parameters are i8* by the writer, so force all pointer to those
      // types here.
      FunctionType *FT =
          cast<FunctionType>(GetAssociatedTypeVariable(CB->getFunctionType()));
      if (isa<InlineAsm>(CB->getCalledOperand()))
        FT = cast<FunctionType>(getUnknownTyped(CB->getFunctionType()));
      else
        TypeRules.push_back(TypeRule::pointsTo(CB->getCalledOperandUse(), FT));
      typeFunctionParams(*CB, FT, 0, true, TypeRules);
    }
  } else if (isa<ExtractElementInst>(&I)) {
    if (!hasPointerType(I.getType()))
      return;
    TypeRules.push_back(TypeRule::propagatesIndirect(I, 0));
  } else if (isa<InsertElementInst>(&I)) {
    if (!hasPointerType(I.getType()))
      return;
    TypeRules.push_back(TypeRule::propagatesIndirect(I, 0));
    TypeRules.push_back(TypeRule::propagatesIndirect(I, 1));
  } else if (isa<ShuffleVectorInst>(&I)) {
    if (!hasPointerType(I.getType()))
      return;
    TypeRules.push_back(TypeRule::propagatesIndirect(I, 0));
    TypeRules.push_back(TypeRule::propagatesIndirect(I, 1));
  }

  // TODO: Handle insertvalue, extractvalue that work with pointers (requires
  // literal struct support)
}

std::pair<Use &, Type *>
SPIRVTypeScavenger::getTypeCheck(Instruction &I, const TypeRule &Rule) {
  auto MakeCheck = [&](Use &U, bool UIndirect, Type *Ty, bool TIndirect) {
    return std::pair<Use &, Type *>(
        U, adjustIndirect(U->getType(), UIndirect, Ty, TIndirect));
  };
  bool LIndirect = Rule.LhsIndirect, RIndirect = Rule.RhsIndirect;
  // If we have typeof(return) == typeof(operand) check, reverse the check for
  // typing rules.
  if (Rule.OpNo == RETURN_OPERAND) {
    Use &U = *cast<Use *>(Rule.Target);
    Type *Ty = getTypeAfterRules(&I);
    return MakeCheck(U, RIndirect, Ty, LIndirect);
  }
  Type *TargetTy;
  if (auto *UsedTy = dyn_cast<Type *>(Rule.Target)) {
    TargetTy = UsedTy;
  } else {
    TargetTy = getTypeAfterRules(cast<Use *>(Rule.Target)->get());
  }
  Use &U = I.getOperandUse(Rule.OpNo);
  return MakeCheck(U, LIndirect, TargetTy, RIndirect);
}

void SPIRVTypeScavenger::correctUseTypes(Instruction &I) {
  // This represents the types of all pointer-valued operands of the
  // instruction.
  SmallVector<TypeRule, 4> TypeRules;
  getTypeRules(I, TypeRules);

  if (!TypeRules.empty())
    LLVM_DEBUG(dbgs() << "Typing uses of " << I << "\n");

  // Now that we've collected all the pointer-valued operands in the
  // instruction, go through and insert bitcasts for any operands that have the
  // wrong type, fix any deferred types whose types are now known, and merge any
  // deferred types that need to have the same type.
  IRBuilder<NoFolder> Builder(&I);

  for (auto &Rule : TypeRules) {
    // No type checking needs to happen for a returns-type rule, since there's
    // no operands of this instruction to check.
    if (Rule.OpNo == RETURN_OPERAND && isa<Type *>(Rule.Target))
      continue;
    auto [U, UsedTy] = getTypeCheck(I, Rule);
    Type *SourceTy = getTypeAfterRules(U);

    // If we're handling a PHI node, we need to insert in the basic block that
    // the value comes in from, not immediately before this instruction.
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      BasicBlock *SourceBlock = Phi->getIncomingBlock(U);
      Builder.SetInsertPoint(SourceBlock->getTerminator());
    }

    bool CanUnify = unifyType(SourceTy, UsedTy);
    LLVM_DEBUG(dbgs() << "  " << *SourceTy << " == " << *UsedTy << "? "
                      << (CanUnify ? "yes" : "no") << "\n");
    if (!CanUnify) {
      LLVM_DEBUG({
        dbgs() << "  Inserting bitcast of ";
        U->printAsOperand(dbgs(), true,
                          I.getParent()->getParent()->getParent());
        dbgs() << "\n";
      });
      Value *CastedValue =
          Builder.Insert(CastInst::CreatePointerCast(U, U->getType()));
      DeducedTypes[CastedValue] = UsedTy;
      U.set(CastedValue);
    }
  }
}

Type *SPIRVTypeScavenger::allocateTypeVariable(Type *Base) {
  LLVMContext &Ctx = Base->getContext();
  return mutateType(Base, [&](unsigned AS) {
    unsigned VarIndex = TypeVariables.size();
    UnifiedTypeVars.grow(VarIndex + 1);
    TypeVariables.push_back(nullptr);
    Type *InnerTy = TargetExtType::get(Ctx, "typevar", {}, {VarIndex});
    return TypedPointerType::get(InnerTy, AS);
  });
}

FunctionType *SPIRVTypeScavenger::getFunctionType(Function *F) {
  TypedPointerType *Ty =
      cast<TypedPointerType>(substituteTypeVariables(DeducedTypes[F]));
  return cast<FunctionType>(Ty->getElementType());
}

Type *SPIRVTypeScavenger::getScavengedType(Value *V) {
  Type *Ty = V->getType();
  if (!hasPointerType(Ty))
    return Ty;

  // If we get a null/undef/poison value (this should be rare, but it can
  // happen if you use, e.g., store ptr null, ptr %val), then assume the result
  // should be an i8. This aligns with the use in the original deduction.
  if (doesNotImplyType(V))
    return getUnknownTyped(Ty);

  auto It = DeducedTypes.find(V);
  if (It != DeducedTypes.end()) {
    return substituteTypeVariables(It->second);
  }

  assert(
      (!isa<Instruction>(V) || !cast<Instruction>(V)->getParent()) &&
      !isa<Argument>(V) && !isa<GlobalValue>(V) &&
      "Global values, arguments, and instructions should all have been typed.");

  // A constant array or constant vector that is used as a global variable
  // initializer should get the type of that global variable.
  if (isa<ConstantArray>(V) || isa<ConstantVector>(V)) {
    for (User *U : V->users()) {
      if (isa<GlobalVariable>(U)) {
        return cast<TypedPointerType>(getScavengedType(U))->getElementType();
      }
    }
  }

  return getUnknownTyped(Ty);
}
