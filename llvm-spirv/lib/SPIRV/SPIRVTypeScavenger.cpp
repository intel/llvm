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
// IR for the output SPIR-V file after LLVM IR completes its transition to
// opaque pointers.
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

#define DEBUG_TYPE "type-scavenger"

using namespace llvm;

void SPIRVTypeScavenger::typeModule(Module &M) {
  // If typed pointers are in effect, we need to do nothing here.
  if (M.getContext().supportsTypedPointers())
    return;

  // Try to fill in any known types for function parameters.
  for (auto &F : M.functions()) {
    deduceFunctionType(F);
  }

  // Collect types for all pertinent values in the module.
  for (auto &F : M.functions()) {
    for (Argument &Arg : F.args())
      if (Arg.getType()->isPointerTy())
        computePointerElementType(&Arg);
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (I.getType()->isPointerTy())
          computePointerElementType(&I);
        correctUseTypes(I);
      }
    }
  }

  // Go through all of the types we have collected, and if any are still
  // deferred, assign them a fallback i8* type.
  Type *Int8Ty = Type::getInt8Ty(M.getContext());
  for (const auto &Pair : DeducedTypes) {
    if (auto *Untyped = dyn_cast<DeferredType *>(Pair.second)) {
      LLVM_DEBUG(dbgs() << "No inferrable type for " << *Pair.first << "\n");
      fixType(*Untyped, Int8Ty);
      DeducedTypes[Pair.first] = Int8Ty;
    }
  }
  return;
}

static Type *getPointerUseType(Function *F, Op Opcode, unsigned ArgNo) {
  switch (Opcode) {
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
    if (ArgNo == 0)
      return F->getReturnType();
    return nullptr;
  case OpAtomicStore:
    if (ArgNo == 0)
      return F->getArg(3)->getType();
    return nullptr;
  default:
    return nullptr;
  }
}

bool SPIRVTypeScavenger::typeIntrinsicCall(
    CallBase &CB, SmallVectorImpl<std::pair<unsigned, DeducedType>> &ArgTys) {
  Function *TargetFn = CB.getCalledFunction();
  assert(TargetFn && TargetFn->isDeclaration() &&
         "Call is not an intrinsic function call");
  LLVMContext &Ctx = TargetFn->getContext();

  if (auto IntrinID = TargetFn->getIntrinsicID()) {
    switch (IntrinID) {
    case Intrinsic::memcpy: {
      // First two parameters are pointers, but it may be any pointer type.
      DeducedType MemcpyTy = new DeferredType;
      ArgTys.emplace_back(0, MemcpyTy);
      ArgTys.emplace_back(1, MemcpyTy);
      break;
    }
    case Intrinsic::memset:
      ArgTys.emplace_back(0, Type::getInt8Ty(Ctx));
      break;
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::invariant_start:
      // These intrinsics were stored as i8* as typed pointers, and the SPIR-V
      // writer will expect these to be i8*, even if they can be any pointer
      // type.
      ArgTys.emplace_back(1, Type::getInt8Ty(Ctx));
      break;
    case Intrinsic::invariant_end:
      // This is like invariant_start with an extra string parameter in the
      // beginning (so the pointer object moves to argument two).
      ArgTys.emplace_back(0, Type::getInt8Ty(Ctx));
      ArgTys.emplace_back(2, Type::getInt8Ty(Ctx));
      break;
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
      // The first parameter of these is an i8*.
      ArgTys.emplace_back(0, Type::getInt8Ty(Ctx));
      [[fallthrough]];
    case Intrinsic::annotation:
      // Second and third parameters are strings, which should be constants
      // for global variables. Nominally, this is i8*, but we specifically
      // *do not* want to insert bitcast instructions (they need to remain
      // global constants).
      break;
    case Intrinsic::stacksave:
      // TODO: support return type.
      break;
    case Intrinsic::stackrestore:
      ArgTys.emplace_back(0, Type::getInt8Ty(Ctx));
      break;
    case Intrinsic::instrprof_cover:
    case Intrinsic::instrprof_increment:
    case Intrinsic::instrprof_increment_step:
    case Intrinsic::instrprof_value_profile:
      // llvm.instrprof.* intrinsics are not supported
      ArgTys.emplace_back(0, Type::getInt8Ty(Ctx));
      break;
    // TODO: handle masked gather/scatter intrinsics. This requires support
    // for vector-of-pointers in the type scavenger.
    default:
      return false;
    }
  } else if (TargetFn->getName().startswith("_Z18__spirv_ocl_printf")) {
    ArgTys.emplace_back(0, Type::getInt8Ty(Ctx));
  } else
    return false;

  return true;
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
  SmallVector<Argument *, 8> PointerArgs;
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy())
      PointerArgs.push_back(&Arg);
  }

  // Get any arguments from attributes where possible.
  for (Argument *Arg : PointerArgs) {
    Type *Ty = getParamType(F.getAttributes(), Arg->getArgNo());
    if (Ty)
      DeducedTypes[Arg] = Ty;
  }

  // At this point, anything that we can get definitively correct is going to
  // come from declarations of builtins. If we have the actual implementation of
  // the function available, we should try to recover types from the function
  // definition itself. By early returning here, we ensure that remaining
  // arguments will get deferred types that will follow the regular typing
  // process.
  if (!F.isDeclaration())
    return;

  // Recover known information from known SPIR-V builtin operations represented
  // as functions.
  StringRef DemangledName;
  if (oclIsBuiltin(F.getName(), DemangledName) ||
      isDecoratedSPIRVFunc(&F, DemangledName)) {
    Op OC = getSPIRVFuncOC(DemangledName);
    if (OC != OpNop) {
      for (Argument *Arg : PointerArgs) {
        Type *PointeeTy = getPointerUseType(&F, OC, Arg->getArgNo());
        if (PointeeTy) {
          DeducedTypes[Arg] = PointeeTy;
          LLVM_DEBUG(dbgs()
                     << "Arg " << Arg->getArgNo() << " of " << F.getName()
                     << " has type " << *PointeeTy << "\n");
        }
      }
    }
  }

  // If the function is a mangled name, try to recover types from the Itanium
  // name mangling.
  if (F.getName().startswith("_Z")) {
    SmallVector<Type *, 8> ParamTypes;
    if (!getParameterTypes(&F, ParamTypes)) {
      return;
    }
    for (Argument *Arg : PointerArgs) {
      if (auto *Ty = dyn_cast<TypedPointerType>(ParamTypes[Arg->getArgNo()])) {
        DeducedTypes[Arg] = Ty->getElementType();
        LLVM_DEBUG(dbgs() << "Arg " << Arg->getArgNo() << " of " << F.getName()
                          << " has type " << *Ty->getElementType() << "\n");
      }
    }
  }
}

/// Certain constant types (null, undef, and poison) will get their type from
/// the use of the constant. We discover the type of the use by inserting a
/// synthetic bitcast instruction before the use. For these types, we need to
/// have special handling in a few places, and this indicates that it needs to
/// be done.
static bool doesNotImplyType(Value *V) {
  return isa<ConstantPointerNull>(V) || isa<UndefValue>(V);
}

SPIRVTypeScavenger::DeducedType
SPIRVTypeScavenger::computePointerElementType(Value *V) {
  assert(V->getType()->isPtrOrPtrVectorTy() &&
         "Trying to get the pointer type of a non-pointer value?");

  // Don't try to store null, undef, or poison in our type map. We'll call these
  // i8* by default; if any use has a different type, a bitcast will be added
  // later.
  if (doesNotImplyType(V)) {
    return Type::getInt8Ty(V->getContext());
  }

  // Check if we've already deduced a type for the value.
  DeducedType &Ty = DeducedTypes[V];
  if (Ty) {
    return Ty;
  }

  assert(!is_contained(VisitStack, V) && "Found cycle in type scavenger");
  VisitStack.push_back(V);

  // There are basically three categories of pointer-typed values:
  // 1. Values that have a well-defined pointee type (e.g., alloca). Return the
  //    known type from this method.
  // 2. Values that have no intrinsic type (e.g., inttoptr). A new deferred
  //    type construct will be created that will allow a use to identify the
  //    type instead.
  // 3. Values that propagate their source type (e.g., phi). This wil return
  //    the type of their source argument, whether it is deferred or known.

  // This lambda does the logic to propagate the third category.
  auto PropagateType = [&](Value *Source) -> DeducedType {
    // If the source argument is null, undef, or poison, then consider the
    // propagation to be untyped. This will fall through to the case where we
    // construct a nw
    if (doesNotImplyType(Source))
      return nullptr;

    DeducedType SourceTy = computePointerElementType(Source);
    if (auto *Deferred = dyn_cast<DeferredType *>(SourceTy)) {
      LLVM_DEBUG(dbgs() << *Source << " will receive the same type as " << *V
                        << "\n");
      Deferred->Values.push_back(V);
    }
    return SourceTy;
  };

  // These values have a natural pointer type (category 1).
  if (auto *GV = dyn_cast<GlobalValue>(V))
    Ty = GV->getValueType();
  else if (auto *Alloca = dyn_cast<AllocaInst>(V))
    Ty = Alloca->getAllocatedType();
  else if (auto *GEP = dyn_cast<GEPOperator>(V))
    Ty = GEP->getResultElementType();

  // These values have no intrinsic type (category 2).
  else if (isa<IntToPtrInst>(V) || isa<BitCastInst>(V))
    Ty = nullptr;

  // These values propagate the source type (category 3).
  else if (auto *AS = dyn_cast<AddrSpaceCastInst>(V))
    Ty = PropagateType(AS->getPointerOperand());
  else if (auto *Freeze = dyn_cast<FreezeInst>(V))
    Ty = PropagateType(Freeze->getOperand(0));
  // Yes, atomicrmw xchg can exchange a ptr type. This will also be considered
  // to propagate the source type.
  else if (auto *AI = dyn_cast<AtomicRMWInst>(V))
    Ty = PropagateType(AI->getValOperand());

  // Selects and phis propagate types as well. Only investigate one of the
  // sources here as the type of the operation: if the primary operand has a
  // deferred type and a secondary operand has a known type, we'll discover that
  // when we handle uses anyways.
  else if (auto *Select = dyn_cast<SelectInst>(V))
    Ty = PropagateType(Select->getTrueValue());
  else if (auto *Phi = dyn_cast<PHINode>(V)) {
    // If we specifically tried the first argument (or any particular argument),
    // we could end up in a situation where we get caught in a cycle:
    // %a = phi(%b, %c)
    // %b = phi(%a, %d)
    // So pick the first argument that whose type we are not trying to compute
    // right now. In the rare case that we have an unreachable block, we could
    // exhaust all possible options, in which case we'll fall through to having
    // an unknown type.
    for (Value *Arg : Phi->incoming_values()) {
      if (!is_contained(VisitStack, Arg)) {
        Ty = PropagateType(Arg);
        break;
      }
    }
  }

  else if (auto *Arg = dyn_cast<Argument>(V)) {
    // Check for an sret/byval/etc. attribute on the argument. If it doesn't
    // have one, then it will return null. There are other cases where we can
    // pre-fill the type of an argument, but that is handled in an earlier
    // pre-pass.
    unsigned ArgNo = Arg->getArgNo();
    Ty = getParamType(Arg->getParent()->getAttributes(), ArgNo);
  } else if (auto *CB = dyn_cast<CallBase>(V)) {
    // TODO: Handle return types properly.
    Ty = Type::getInt8Ty(CB->getContext());
  } else {
    // TODO: handle pointer-valued extractvalue, which probably comes from
    // cmpxchg or inlineasm.
    LLVM_DEBUG(
        dbgs()
        << "Value " << *V << " is not a known type of "
        << "pointer-valued instruction, this logic is probably wrong!\n");
  }

  // If we haven't gotten a type at this point, we need to construct a new
  // deferred type to handle this value. This also considers cases where we
  // were trying to propagate a null constant.
  if (!Ty) {
    LLVM_DEBUG(dbgs() << "Value " << *V
                      << " has no known type, creating a new type for it\n");
    DeferredType *Deferred = new DeferredType;
    Deferred->Values.push_back(V);
    Ty = Deferred;
  }

  VisitStack.pop_back();
  return Ty;
}

void SPIRVTypeScavenger::correctUseTypes(Instruction &I) {
  // This represents the types of all pointer-valued operands of the
  // instruction.
  SmallVector<std::pair<unsigned, DeducedType>, 4> PointerOperands;

  // For instructions which operate with memory (e.g., load, store), this is the
  // value whose type will determine the type of the operand. When the memory
  // type is just a generic `ptr` type, this will be used to generate a
  // pointer-to-pointer-to-something type.
  auto GetMemoryType = [&](Value *V) -> DeducedType {
    if (V->getType()->isPointerTy())
      return V;
    return V->getType();
  };

  // Basic instructions that have a clearly fixed type.
  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    PointerOperands.emplace_back(GetElementPtrInst::getPointerOperandIndex(),
                                 GEP->getSourceElementType());
  } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
    PointerOperands.emplace_back(LoadInst::getPointerOperandIndex(),
                                 GetMemoryType(LI));
  } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
    PointerOperands.emplace_back(StoreInst::getPointerOperandIndex(),
                                 GetMemoryType(SI->getValueOperand()));
  } else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(&I)) {
    PointerOperands.emplace_back(AtomicCmpXchgInst::getPointerOperandIndex(),
                                 GetMemoryType(AI->getCompareOperand()));
  } else if (auto *AI = dyn_cast<AtomicRMWInst>(&I)) {
    PointerOperands.emplace_back(AtomicRMWInst::getPointerOperandIndex(),
                                 GetMemoryType(AI->getValOperand()));
  } else if (auto *CI = dyn_cast<ICmpInst>(&I)) {
    // icmp can compare pointers. If it isn't, ignore the instruction.
    if (!CI->getOperand(0)->getType()->isPointerTy())
      return;

    // The two pointer operands should have the same type.
    PointerOperands.emplace_back(1,
                                 computePointerElementType(CI->getOperand(0)));
  } else if (auto *SI = dyn_cast<SelectInst>(&I)) {
    if (!SI->getType()->isPointerTy())
      return;

    // Both selected values should have the same type as the result.
    DeducedType Ty = computePointerElementType(SI);
    PointerOperands.emplace_back(1, Ty);
    PointerOperands.emplace_back(2, Ty);
  } else if (auto *Phi = dyn_cast<PHINode>(&I)) {
    if (!Phi->getType()->isPointerTy())
      return;

    DeducedType Ty = computePointerElementType(Phi);
    for (Use &U : Phi->incoming_values()) {
      PointerOperands.emplace_back(U.getOperandNo(), Ty);
    }
  } else if (isa<FreezeInst>(&I) || isa<AddrSpaceCastInst>(&I)) {
    if (!I.getType()->isPointerTy())
      return;
    PointerOperands.emplace_back(0, computePointerElementType(&I));
  } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
    if (!RI->getReturnValue() ||
        !RI->getReturnValue()->getType()->isPointerTy())
      return;
    // TODO: Handle return types properly.
    PointerOperands.emplace_back(0, Type::getInt8Ty(I.getContext()));
  } else if (auto *CB = dyn_cast<CallBase>(&I)) {
    PointerOperands.emplace_back(CB->getCalledOperandUse().getOperandNo(),
                                 CB->getFunctionType());
    // If we have an identified function for the call instruction, map the
    // arguments we pass in to the argument requirements of the function.
    if (Function *F = CB->getCalledFunction()) {
      if (!F->isDeclaration() || !typeIntrinsicCall(*CB, PointerOperands)) {
        for (Use &U : CB->args()) {
          // If we're calling a var-arg method, we have more operands than the
          // function has parameters. Bail out if we hit that point.
          unsigned ArgNo = CB->getArgOperandNo(&U);
          if (ArgNo >= F->arg_size())
            break;
          if (U->getType()->isPointerTy())
            PointerOperands.emplace_back(
                U.getOperandNo(), computePointerElementType(F->getArg(ArgNo)));
        }
      }
    }
  }

  // TODO: Handle insertvalue instructions that insert pointers.

  // Now that we've collected all the pointer-valued operands in the
  // instruction, go through and insert bitcasts for any operands that have the
  // wrong type, fix any deferred types whose types are now known, and merge any
  // deferred types that need to have the same type.
  IRBuilder<NoFolder> Builder(&I);

  for (auto &Pair : PointerOperands) {
    Use &U = I.getOperandUse(Pair.first);
    DeducedType UsedTy = Pair.second;
    DeducedType SourceTy = computePointerElementType(U);

    // If we're handling a PHI node, we need to insert in the basic block that
    // the value comes in from, not immediately before this instruction.
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      BasicBlock *SourceBlock = Phi->getIncomingBlock(U);
      Builder.SetInsertPoint(SourceBlock->getTerminator());
    }

    auto InsertCast = [&]() {
      if (isa<Type *>(UsedTy)) {
        LLVM_DEBUG(dbgs() << "Inserting bitcast of " << *U.get()
                          << " to change its type to " << *cast<Type *>(UsedTy)
                          << " because of use in " << *U.getUser() << "\n");
      } else {
        LLVM_DEBUG(dbgs() << "Inserting bitcast of " << *U.get()
                          << " for indirect pointer use of "
                          << *cast<Value *>(UsedTy) << " because of use in "
                          << *U.getUser() << "\n");
      }
      Value *CastedValue =
          Builder.Insert(CastInst::CreatePointerCast(U, U->getType()));
      DeducedTypes[CastedValue] = UsedTy;
      U.set(CastedValue);
    };

    // This handles the scenario where a deferred type gets resolved to a fixed
    // type during handling of this instruction, and another operand is using
    // the same deferred type later in the instruction.
    auto ReplaceTypeInOperands = [&](DeducedType From, DeducedType To) {
      for (auto &ReplacePair : PointerOperands) {
        if (ReplacePair.second == From)
          ReplacePair.second = To;
      }
    };

    if (isa<Value *>(UsedTy)) {
      // When the use is of an indirect-pointer type, insert a bitcast to the
      // use type only for this use. This prevents indirect pointers from
      // generally leaking into more of the type system and causing potential
      // issues.
      InsertCast();
    } else if (auto *FixedTy = dyn_cast<Type *>(SourceTy)) {
      if (auto *FixedUseTy = dyn_cast<Type *>(UsedTy)) {
        // Both source and use type are fixed -> insert a bitcast are different.
        if (FixedTy != FixedUseTy) {
          InsertCast();
        }
      } else if (auto *DeferredUseTy = dyn_cast<DeferredType *>(UsedTy)) {
        // Source type is fixed, use type is deferred: set the deferred type to
        // the fixed type.
        ReplaceTypeInOperands(DeferredUseTy, FixedTy);
        fixType(*DeferredUseTy, FixedTy);
      }
    } else if (auto *DeferredTy = dyn_cast<DeferredType *>(SourceTy)) {
      if (auto *FixedUseTy = dyn_cast<Type *>(UsedTy)) {
        // Source type is fixed, use type is deferred: set the deferred type to
        // the fixed type.
        ReplaceTypeInOperands(DeferredTy, FixedUseTy);
        fixType(*DeferredTy, FixedUseTy);
      } else if (auto *DeferredUseTy = dyn_cast<DeferredType *>(UsedTy)) {
        // If they're both deferred, merge the two types together.
        ReplaceTypeInOperands(DeferredUseTy, DeferredTy);
        mergeType(DeferredTy, DeferredUseTy);
      }
    }
  }
}

void SPIRVTypeScavenger::fixType(DeferredType &Ty, Type *ActualTy) {
  for (Value *V : Ty.Values) {
    LLVM_DEBUG(dbgs() << "Inferred type of " << *V << " to be " << *ActualTy
                      << "\n");
    DeducedTypes[V] = ActualTy;
  }
  delete &Ty;
}

void SPIRVTypeScavenger::mergeType(DeferredType *Ty1, DeferredType *Ty2) {
  // It's possible we're trying to merge the same type into itself.
  if (Ty1 == Ty2)
    return;

  for (Value *V : Ty2->Values) {
    DeducedTypes[V] = Ty1;
    Ty1->Values.push_back(V);
  }
  delete Ty2;
}

SPIRVTypeScavenger::PointeeType
SPIRVTypeScavenger::getPointerElementType(Value *V) {
  PointerType *Ty = dyn_cast<PointerType>(V->getType());
  assert(Ty && "Non-pointer types don't have pointee types");

  if (!Ty->isOpaquePointerTy())
    return Ty->getNonOpaquePointerElementType();

  // Global values have a natural pointee type that we can use.
  if (auto *GV = dyn_cast<GlobalValue>(V))
    return GV->getValueType();

  // If we get a null/undef/poison value (this should be rare, but it can
  // happen if you use, e.g., store ptr null, ptr %val), then assume the result
  // should be an i8. This aligns with the use in the original deduction.
  if (doesNotImplyType(V)) {
    return Type::getInt8Ty(V->getContext());
  }

  // If it's a constant expression, we won't have a type for it. Constant
  // expressions are currently translated via converting them to instructions
  // without a basic block.
  bool IsFromConstantExpr =
      isa<ConstantExpr>(V) ||
      (isa<Instruction>(V) && !cast<Instruction>(V)->getParent());
  (void)IsFromConstantExpr;
  auto It = DeducedTypes.find(V);
  assert((It != DeducedTypes.end() || IsFromConstantExpr) &&
         "How have we not typed the value?");
  if (It != DeducedTypes.end()) {
    if (auto *Ty = dyn_cast<Type *>(It->second))
      return Ty;
    if (auto *ValTy = dyn_cast<Value *>(It->second))
      return ValTy;
    llvm_unreachable("Deferred types should have been resolved before now");
  }

  return Type::getInt8Ty(V->getContext());
}

Type *SPIRVTypeScavenger::getArgumentPointerElementType(Function *F,
                                                        unsigned ArgNo) {
  return cast<Type *>(getPointerElementType(F->getArg(ArgNo)));
}
