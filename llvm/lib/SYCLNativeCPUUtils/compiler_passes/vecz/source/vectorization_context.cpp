// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "vectorization_context.h"

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/group_collective_helpers.h>
#include <compiler/utils/mangling.h>
#include <compiler/utils/pass_functions.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/AtomicOrdering.h>
#include <llvm/Target/TargetMachine.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include <algorithm>
#include <cassert>
#include <optional>

#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "llvm_helpers.h"
#include "memory_operations.h"
#include "transform/packetization_helpers.h"
#include "vectorization_helpers.h"
#include "vectorization_unit.h"
#include "vecz/vecz_choices.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;
using namespace vecz;

STATISTIC(VeczContextFailBuiltin,
          "Context: builtins with no vector equivalent [ID#V84]");
STATISTIC(VeczContextFailScalarizeCall,
          "Context: non-scalarizable vector builtin [ID#V86]");

/// @brief Prefix used to distinguish internal vecz builtins from OpenCL
/// builtins and user functions.
const char *VectorizationContext::InternalBuiltinPrefix = "__vecz_b_";

VectorizationContext::VectorizationContext(llvm::Module &target,
                                           TargetInfo &vti,
                                           compiler::utils::BuiltinInfo &bi)
    : VTI(vti), Module(target), BI(bi), DL(&Module.getDataLayout()) {}

TargetTransformInfo VectorizationContext::getTargetTransformInfo(
    Function &F) const {
  auto *const TM = targetInfo().getTargetMachine();
  if (TM) {
    return TM->getTargetTransformInfo(F);
  } else {
    return TargetTransformInfo(F.getParent()->getDataLayout());
  }
}

VectorizationUnit *VectorizationContext::getActiveVU(const Function *F) const {
  const auto I = ActiveVUs.find(F);
  if (I == ActiveVUs.end()) {
    return nullptr;
  }
  VectorizationUnit *VU = I->second;
  assert(VU->vectorizedFunction() == F);
  return VU;
}

compiler::utils::BuiltinInfo &VectorizationContext::builtins() { return BI; }

const compiler::utils::BuiltinInfo &VectorizationContext::builtins() const {
  return BI;
}

VectorizationUnit *VectorizationContext::createVectorizationUnit(
    llvm::Function &F, ElementCount VF, unsigned Dimension,
    const VectorizationChoices &Ch) {
  KernelUnits.push_back(
      std::make_unique<VectorizationUnit>(F, VF, Dimension, *this, Ch));
  return KernelUnits.back().get();
}

bool VectorizationContext::isVector(const Instruction &I) {
  if (I.getType()->isVectorTy()) {
    return true;
  }
  for (const Use &op : I.operands()) {
    if (op->getType()->isVectorTy()) {
      return true;
    }
  }
  return false;
}

bool VectorizationContext::canExpandBuiltin(const Function *ScalarFn) const {
  // Builtins that return no value must have side-effects.
  if (ScalarFn->getReturnType()->isVoidTy()) {
    return false;
  }
  for (const Argument &Arg : ScalarFn->args()) {
    // Most builtins that take pointers have side-effects. Be conservative.
    if (Arg.getType()->isPointerTy()) {
      return false;
    }
  }
  return true;
}

VectorizationResult &VectorizationContext::getOrCreateBuiltin(
    llvm::Function &F, unsigned SimdWidth) {
  compiler::utils::BuiltinInfo &BI = builtins();
  const auto Cached = VectorizedBuiltins.find(&F);
  if (Cached != VectorizedBuiltins.end()) {
    const auto Found = Cached->second.find(SimdWidth);
    if (Found != Cached->second.end()) {
      return Found->second;
    }
  }

  const auto Builtin = BI.analyzeBuiltin(F);

  // Try to find a vector equivalent for the builtin.
  Function *const VectorCallee =
      isInternalBuiltin(&F)
          ? getInternalVectorEquivalent(&F, SimdWidth)
          : BI.getVectorEquivalent(Builtin, SimdWidth, &Module);

  auto &result = VectorizedBuiltins[&F][SimdWidth];
  if (!VectorCallee) {
    ++VeczContextFailBuiltin;
    return result;
  }

  result.func = VectorCallee;

  // Gather information about the function's arguments.
  const auto Props = Builtin.properties;
  unsigned i = 0;
  for (const Argument &Arg : F.args()) {
    Type *pointerRetPointeeTy = nullptr;
    VectorizationResult::Arg::Kind kind = VectorizationResult::Arg::SCALAR;

    if (Arg.getType()->isPointerTy()) {
      pointerRetPointeeTy =
          compiler::utils::getPointerReturnPointeeTy(F, Props);
      kind = VectorizationResult::Arg::POINTER_RETURN;
    } else {
      kind = VectorizationResult::Arg::VECTORIZED;
    }
    result.args.emplace_back(kind, VectorCallee->getArg(i)->getType(),
                             pointerRetPointeeTy);
    i++;
  }
  return result;
}

VectorizationResult VectorizationContext::getVectorizedFunction(
    Function &callee, ElementCount factor) {
  VectorizationResult result;
  if (factor.isScalable()) {
    // We can't vectorize builtins by a scalable factor yet.
    return result;
  }

  auto simdWidth = factor.getFixedValue();
  if (auto *vecTy = dyn_cast<FixedVectorType>(callee.getReturnType())) {
    const auto Builtin = BI.analyzeBuiltin(callee);
    Function *scalarEquiv = builtins().getScalarEquivalent(Builtin, &Module);
    if (!scalarEquiv) {
      ++VeczContextFailScalarizeCall;
      return VectorizationResult();
    }

    auto scalarWidth = vecTy->getNumElements();

    result = getOrCreateBuiltin(*scalarEquiv, simdWidth * scalarWidth);
  } else {
    result = getOrCreateBuiltin(callee, simdWidth);
  }
  return result;
}

bool VectorizationContext::isInternalBuiltin(const Function *F) {
  return F->getName().starts_with(VectorizationContext::InternalBuiltinPrefix);
}

Function *VectorizationContext::getOrCreateInternalBuiltin(StringRef Name,
                                                           FunctionType *FT) {
  Function *F = Module.getFunction(Name);
  if (!F && FT) {
    F = dyn_cast_or_null<Function>(
        Module.getOrInsertFunction(Name, FT).getCallee());
    if (F) {
      // Set some default attributes on the function.
      // We never use exceptions
      F->addFnAttr(Attribute::NoUnwind);
      // Recursion is not supported in ComputeMux
      F->addFnAttr(Attribute::NoRecurse);
    }
  }

  return F;
}

Function *VectorizationContext::getOrCreateMaskedFunction(CallInst *CI) {
  Function *F = CI->getCalledFunction();
  if (!F) {
    F = dyn_cast<Function>(CI->getCalledOperand()->stripPointerCasts());
  }
  VECZ_FAIL_IF(!F);  // TODO: CA-1505: Support indirect function calls.
  LLVMContext &ctx = F->getContext();

  // We will handle printf statements, but handling every possible vararg
  // function can become a bit too complex, among other things because name
  // mangling with arbitrary types can become a bit complex. printf is the only
  // vararg OpenCL builtin, so only user functions are affected by this.
  const bool isVarArg = F->isVarArg();
  VECZ_FAIL_IF(isVarArg && F->getName() != "printf");
  // Copy the argument types. This is done from the CallInst instead of the
  // called Function because the called Function might be a VarArg function, in
  // which case we need to create the wrapper with the expanded argument list.
  SmallVector<Type *, 8> argTys;
  for (const auto &U : CI->args()) {
    argTys.push_back(U->getType());
  }
  AttributeList fnAttrs = F->getAttributes();
  unsigned firstImmArg;
  const bool hasImmArg =
      F->isIntrinsic() &&
      fnAttrs.hasAttrSomewhere(Attribute::ImmArg, &firstImmArg);
  if (hasImmArg) {
    firstImmArg -= AttributeList::FirstArgIndex;
    // We can only handle a single `i1` `Immarg` parameter. If we outgrow this
    // limitation we need a different approach to the single inner branch
    int count = 0;
    for (unsigned i = firstImmArg, n = argTys.size(); i < n; ++i) {
      if (!fnAttrs.hasAttributeAtIndex(AttributeList::FirstArgIndex + i,
                                       Attribute::ImmArg)) {
        continue;
      }
      // We only support one ImmArg or i1 type
      if (count++ || argTys[i] != Type::getInt1Ty(ctx)) {
        return nullptr;
      }
      fnAttrs = fnAttrs.removeAttributeAtIndex(ctx, i, Attribute::ImmArg);
    }
  }
  // Add one extra argument for the mask
  argTys.push_back(Type::getInt1Ty(ctx));
  // Generate the function name
  compiler::utils::NameMangler mangler(&ctx);
  const SmallVector<compiler::utils::TypeQualifiers, 8> quals(
      argTys.size(), compiler::utils::TypeQualifiers());
  std::string newFName;
  raw_string_ostream O(newFName);
  O << VectorizationContext::InternalBuiltinPrefix << "masked_" << F->getName();
  // We need to mangle the names of the vararg masked functions, since we will
  // generate different masked functions for invocations with different argument
  // types. For non-vararg functions, we don't need the mangling so we skip it.
  if (isVarArg) {
    O << "_";
    for (auto T : argTys) {
      VECZ_FAIL_IF(!mangler.mangleType(
          O, T,
          compiler::utils::TypeQualifiers(compiler::utils::eTypeQualNone)));
    }
  }
  O.flush();
  // Check if we have a masked version already
  auto maskedVersion = MaskedVersions.find(newFName);
  if (maskedVersion != MaskedVersions.end()) {
    LLVM_DEBUG(dbgs() << "vecz: Found existing masked function " << newFName
                      << "\n");
    return maskedVersion->second;
  }
  // Create the function type
  FunctionType *newFunctionTy =
      FunctionType::get(F->getReturnType(), argTys, false);
  Function *newFunction = Function::Create(
      newFunctionTy, GlobalValue::PrivateLinkage, newFName, F->getParent());
  const CallingConv::ID cc = CI->getCallingConv();
  LLVM_DEBUG(dbgs() << "vecz: Created masked function " << newFName << "\n");

  // Create the function's basic blocks
  BasicBlock *entryBlock = BasicBlock::Create(ctx, "entry", newFunction);
  BasicBlock *activeBlock = BasicBlock::Create(ctx, "active", newFunction);
  BasicBlock *mergeBlock = BasicBlock::Create(ctx, "exit", newFunction);

  // Create a new call instruction to call the masked function
  SmallVector<Value *, 8> CIArgs;
  for (Value &arg : newFunction->args()) {
    CIArgs.push_back(&arg);
  }
  // Remove the mask argument
  CIArgs.pop_back();

  FunctionType *FTy = CI->getFunctionType();
  const AttributeList callAttrs = CI->getAttributes();
  SmallVector<std::pair<Value *, BasicBlock *>, 4> PhiOperands;
  if (hasImmArg) {
    Value *immArg = newFunction->getArg(firstImmArg);
    BasicBlock *const immTrueBB =
        BasicBlock::Create(ctx, "active.imm.1", newFunction, mergeBlock);
    CIArgs[firstImmArg] = ConstantInt::getTrue(ctx);
    CallInst *c0 =
        CallInst::Create(FTy, CI->getCalledOperand(), CIArgs, "", immTrueBB);
    c0->setCallingConv(cc);
    c0->setAttributes(callAttrs);
    BranchInst::Create(mergeBlock, immTrueBB);

    CIArgs[firstImmArg] = ConstantInt::getFalse(ctx);
    // Now the false half
    BasicBlock *const immFalseBB =
        BasicBlock::Create(ctx, "active.imm.0", newFunction, mergeBlock);

    CallInst *c1 =
        CallInst::Create(FTy, CI->getCalledOperand(), CIArgs, "", immFalseBB);
    c1->setCallingConv(cc);
    c1->setAttributes(callAttrs);
    BranchInst::Create(mergeBlock, immFalseBB);
    BranchInst::Create(immTrueBB, immFalseBB, immArg, activeBlock);
    PhiOperands.push_back({c0, immTrueBB});
    PhiOperands.push_back({c1, immFalseBB});

    // Now fix up the new function's signature. It can't be inheriting illegal
    // attributes; only intrinsics may have the `ImmArg` Attribute. The verifier
    // complains loudly otherwise, and then comes into our houses at night, and
    // wrecks up the place...
    for (unsigned i = 0, n = fnAttrs.getNumAttrSets(); i < n; ++i) {
      fnAttrs = fnAttrs.removeAttributeAtIndex(ctx, i, Attribute::ImmArg);
    }
  } else {
    // We are using the called Value instead of F because it might contain
    // a bitcast or something, which makes the function types different.
    CallInst *c =
        CallInst::Create(FTy, CI->getCalledOperand(), CIArgs, "", activeBlock);
    c->setCallingConv(cc);
    c->setAttributes(callAttrs);
    PhiOperands.push_back({c, activeBlock});
    BranchInst::Create(mergeBlock, activeBlock);
  }
  newFunction->setCallingConv(cc);
  newFunction->setAttributes(fnAttrs);

  // Get the last argument (the mask) and use it as our branch predicate as to
  // the live blocks or a no-op
  Value *mask = newFunction->arg_end() - 1;
  BranchInst::Create(activeBlock, mergeBlock, mask, entryBlock);

  Type *returnTy = F->getReturnType();
  if (returnTy != Type::getVoidTy(ctx)) {
    PHINode *result = PHINode::Create(returnTy, 2, "", mergeBlock);
    for (auto &phiOp : PhiOperands) {
      result->addIncoming(phiOp.first, phiOp.second);
    }
    result->addIncoming(getDefaultValue(returnTy), entryBlock);
    ReturnInst::Create(ctx, result, mergeBlock);
  } else {
    ReturnInst::Create(ctx, mergeBlock);
  }

  MaskedVersions.insert(std::make_pair(newFName, newFunction));
  insertMaskedFunction(newFunction, F);
  return newFunction;
}

std::optional<VectorizationContext::MaskedAtomic>
VectorizationContext::isMaskedAtomicFunction(const Function &F) const {
  auto VFInfo = decodeVectorizedFunctionName(F.getName());
  if (!VFInfo) {
    return std::nullopt;
  }
  auto [FnNameStr, VF, Choices] = *VFInfo;

  llvm::StringRef FnName = FnNameStr;
  if (!FnName.consume_front("masked_")) {
    return std::nullopt;
  }
  const bool IsCmpXchg = FnName.consume_front("cmpxchg_");
  if (!IsCmpXchg && !FnName.consume_front("atomicrmw_")) {
    return std::nullopt;
  }
  VectorizationContext::MaskedAtomic AtomicInfo;

  AtomicInfo.VF = VF;
  AtomicInfo.IsVectorPredicated = Choices.vectorPredication();

  if (IsCmpXchg) {
    AtomicInfo.IsWeak = FnName.consume_front("weak_");
  }
  AtomicInfo.IsVolatile = FnName.consume_front("volatile_");

  if (IsCmpXchg) {
    AtomicInfo.BinOp = AtomicRMWInst::BinOp::BAD_BINOP;
  } else {
    if (FnName.consume_front("xchg")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Xchg;
    } else if (FnName.consume_front("add")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Add;
    } else if (FnName.consume_front("sub")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Sub;
    } else if (FnName.consume_front("and")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::And;
    } else if (FnName.consume_front("nand")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Nand;
    } else if (FnName.consume_front("or")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Or;
    } else if (FnName.consume_front("xor")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Xor;
    } else if (FnName.consume_front("max")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Max;
    } else if (FnName.consume_front("min")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::Min;
    } else if (FnName.consume_front("umax")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::UMax;
    } else if (FnName.consume_front("umin")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::UMin;
    } else if (FnName.consume_front("fadd")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::FAdd;
    } else if (FnName.consume_front("fsub")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::FSub;
    } else if (FnName.consume_front("fmax")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::FMax;
    } else if (FnName.consume_front("fmin")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::FMin;
    } else if (FnName.consume_front("uincwrap")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::UIncWrap;
    } else if (FnName.consume_front("udecwrap")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::UDecWrap;
#if LLVM_VERSION_GREATER_EQUAL(20, 0)
    } else if (FnName.consume_front("usubcond")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::USubCond;
    } else if (FnName.consume_front("usubsat")) {
      AtomicInfo.BinOp = AtomicRMWInst::BinOp::USubSat;
#endif
    } else {
      return std::nullopt;
    }
    if (!FnName.consume_front("_")) {
      return std::nullopt;
    }
  }

  if (!FnName.consume_front("align")) {
    return std::nullopt;
  }

  uint64_t Alignment = 0;
  if (FnName.consumeInteger(/*Radix=*/10, Alignment)) {
    return std::nullopt;
  }

  AtomicInfo.Align = Align(Alignment);

  if (!FnName.consume_front("_")) {
    return std::nullopt;
  }

  auto demangleOrdering = [&FnName]() -> std::optional<AtomicOrdering> {
    if (FnName.consume_front("acquire_")) {
      return AtomicOrdering::Acquire;
    } else if (FnName.consume_front("acqrel_")) {
      return AtomicOrdering::AcquireRelease;
    } else if (FnName.consume_front("monotonic_")) {
      return AtomicOrdering::Monotonic;
    } else if (FnName.consume_front("notatomic_")) {
      return AtomicOrdering::NotAtomic;
    } else if (FnName.consume_front("release_")) {
      return AtomicOrdering::Release;
    } else if (FnName.consume_front("seqcst_")) {
      return AtomicOrdering::SequentiallyConsistent;
    } else if (FnName.consume_front("unordered_")) {
      return AtomicOrdering::Unordered;
    } else {
      return std::nullopt;
    }
  };

  if (auto Ordering = demangleOrdering()) {
    AtomicInfo.Ordering = *Ordering;
  } else {
    return std::nullopt;
  }

  if (IsCmpXchg) {
    if (auto Ordering = demangleOrdering()) {
      AtomicInfo.CmpXchgFailureOrdering = Ordering;
    } else {
      return std::nullopt;
    }
  }

  unsigned SyncScopeID = 0;
  if (FnName.consumeInteger(/*Radix=*/10, SyncScopeID)) {
    return std::nullopt;
  }

  AtomicInfo.SyncScope = static_cast<SyncScope::ID>(SyncScopeID);

  if (!FnName.consume_front("_")) {
    return std::nullopt;
  }

  // Note - we just assume the rest of the builtin name is okay, here. It
  // should be mangled types, but vecz builtins use a strange mangling system,
  // purely for uniqueness and not to infer types. Types are always assumed to
  // be inferrable from the function parameters.
  AtomicInfo.PointerTy = F.getFunctionType()->getParamType(0);
  AtomicInfo.ValTy = F.getFunctionType()->getParamType(1);

  return AtomicInfo;
}

Function *VectorizationContext::getOrCreateMaskedAtomicFunction(
    MaskedAtomic &I, const VectorizationChoices &Choices, ElementCount VF) {
  const bool isCmpXchg = I.isCmpXchg();
  LLVMContext &ctx = I.ValTy->getContext();

  SmallVector<Type *, 8> argTys;

  argTys.push_back(I.PointerTy);
  argTys.push_back(I.ValTy);
  if (isCmpXchg) {
    argTys.push_back(I.ValTy);
  }
  // Add one extra argument for the mask, which is always the same length
  // (scalar or vector) as the value type.
  auto *i1Ty = Type::getInt1Ty(ctx);
  auto *maskTy =
      !I.ValTy->isVectorTy()
          ? dyn_cast<Type>(i1Ty)
          : VectorType::get(i1Ty, cast<VectorType>(I.ValTy)->getElementCount());
  argTys.push_back(maskTy);
  if (Choices.vectorPredication()) {
    argTys.push_back(Type::getInt32Ty(ctx));
  }

  std::string maskedFnName;
  raw_string_ostream O(maskedFnName);
  O << (isCmpXchg ? "masked_cmpxchg_" : "masked_atomicrmw_");

  if (I.IsWeak) {
    assert(isCmpXchg && "Bad MaskedAtomic state");
    O << "weak_";
  }

  if (I.IsVolatile) {
    O << "volatile_";
  }

  if (!isCmpXchg) {
#define BINOP_CASE(BINOP, STR) \
  case AtomicRMWInst::BINOP:   \
    O << (STR);                \
    break

    switch (I.BinOp) {
      BINOP_CASE(Xchg, "xchg");
      BINOP_CASE(Add, "add");
      BINOP_CASE(Sub, "sub");
      BINOP_CASE(And, "and");
      BINOP_CASE(Nand, "nand");
      BINOP_CASE(Or, "or");
      BINOP_CASE(Xor, "xor");
      BINOP_CASE(Max, "max");
      BINOP_CASE(Min, "min");
      BINOP_CASE(UMax, "umax");
      BINOP_CASE(UMin, "umin");
      BINOP_CASE(FAdd, "fadd");
      BINOP_CASE(FSub, "fsub");
      BINOP_CASE(FMax, "fmax");
      BINOP_CASE(FMin, "fmin");
      BINOP_CASE(UIncWrap, "uincwrap");
      BINOP_CASE(UDecWrap, "udecwrap");
#if LLVM_VERSION_GREATER_EQUAL(20, 0)
      BINOP_CASE(USubCond, "usubcond");
      BINOP_CASE(USubSat, "usubsat");
#endif
      case llvm::AtomicRMWInst::BAD_BINOP:
        return nullptr;
    }

#undef BINOP_CASE
    O << "_";
  }

  O << "align" << I.Align.value() << "_";

  // Mangle ordering
  auto mangleOrdering = [&O](AtomicOrdering Ordering) {
    switch (Ordering) {
      default:
        O << static_cast<unsigned>(Ordering);
        break;
      case AtomicOrdering::Acquire:
        O << "acquire";
        break;
      case AtomicOrdering::AcquireRelease:
        O << "acqrel";
        break;
      case AtomicOrdering::Monotonic:
        O << "monotonic";
        break;
      case AtomicOrdering::NotAtomic:
        O << "notatomic";
        break;
      case AtomicOrdering::Release:
        O << "release";
        break;
      case AtomicOrdering::SequentiallyConsistent:
        O << "seqcst";
        break;
      case AtomicOrdering::Unordered:
        O << "unordered";
        break;
    }
  };

  mangleOrdering(I.Ordering);
  // Failure Ordering
  if (I.CmpXchgFailureOrdering) {
    O << "_";
    mangleOrdering(*I.CmpXchgFailureOrdering);
  }

  // Syncscope
  O << "_" << static_cast<unsigned>(I.SyncScope) << "_";

  // Mangle types
  compiler::utils::NameMangler mangler(&ctx);
  for (auto *ty : argTys) {
    VECZ_FAIL_IF(!mangler.mangleType(
        O, ty,
        compiler::utils::TypeQualifiers(compiler::utils::eTypeQualNone)));
  }

  maskedFnName =
      getVectorizedFunctionName(maskedFnName, VF, Choices, /*IsBuiltin=*/true);

  Type *maskedFnRetTy = isCmpXchg ? StructType::get(I.ValTy, maskTy) : I.ValTy;

  // Create the function type
  FunctionType *maskedFnTy =
      FunctionType::get(maskedFnRetTy, argTys, /*isVarArg=*/false);

  return getOrCreateInternalBuiltin(maskedFnName, maskedFnTy);
}

namespace {
std::optional<std::tuple<bool, RecurKind, bool>> isSubgroupScan(
    StringRef fnName, Type *const ty) {
  compiler::utils::Lexer L(fnName);
  if (!L.Consume(VectorizationContext::InternalBuiltinPrefix)) {
    return std::nullopt;
  }
  if (!L.Consume("sub_group_scan_")) {
    return std::nullopt;
  }
  const bool isInt = ty->isIntOrIntVectorTy();
  const bool isInclusive = L.Consume("inclusive_");
  if (isInclusive || L.Consume("exclusive_")) {
    StringRef OpKind;
    if (L.ConsumeAlpha(OpKind)) {
      RecurKind opKind;
      if (OpKind == "add") {
        opKind = isInt ? RecurKind::Add : RecurKind::FAdd;
      } else if (OpKind == "min") {
        assert(!isInt && "unexpected internal scan builtin");
        opKind = RecurKind::FMin;
      } else if (OpKind == "max") {
        assert(!isInt && "unexpected internal scan builtin");
        opKind = RecurKind::FMax;
      } else if (OpKind == "smin") {
        opKind = RecurKind::SMin;
      } else if (OpKind == "smax") {
        opKind = RecurKind::SMax;
      } else if (OpKind == "umin") {
        opKind = RecurKind::UMin;
      } else if (OpKind == "umax") {
        opKind = RecurKind::UMax;
      } else if (OpKind == "mul") {
        opKind = isInt ? RecurKind::Mul : RecurKind::FMul;
      } else if (OpKind == "and") {
        opKind = RecurKind::And;
        assert(isInt && "unexpected internal scan builtin");
      } else if (OpKind == "or") {
        opKind = RecurKind::Or;
        assert(isInt && "unexpected internal scan builtin");
      } else if (OpKind == "xor") {
        opKind = RecurKind::Xor;
        assert(isInt && "unexpected internal scan builtin");
      } else {
        return std::nullopt;
      }
      const bool isVP = L.Consume("_vp");
      return std::make_tuple(isInclusive, opKind, isVP);
    }
  }
  return std::nullopt;
}
}  // namespace

bool VectorizationContext::defineInternalBuiltin(Function *F) {
  assert(F->isDeclaration() && "builtin is already defined");

  // Handle masked memory loads and stores.
  if (std::optional<MemOpDesc> Desc = MemOpDesc::analyzeMemOpFunction(*F)) {
    if (Desc->isMaskedMemOp()) {
      return emitMaskedMemOpBody(*F, *Desc);
    }

    // Handle interleaved memory loads and stores.
    if (Desc->isInterleavedMemOp()) {
      return emitInterleavedMemOpBody(*F, *Desc);
    }

    // Handle masked interleaved memory loads and stores
    if (Desc->isMaskedInterleavedMemOp()) {
      return emitMaskedInterleavedMemOpBody(*F, *Desc);
    }

    // Handle scatter stores and gather loads.
    if (Desc->isScatterGatherMemOp()) {
      return emitScatterGatherMemOpBody(*F, *Desc);
    }

    // Handle masked scatter stores and gather loads.
    if (Desc->isMaskedScatterGatherMemOp()) {
      return emitMaskedScatterGatherMemOpBody(*F, *Desc);
    }
  }

  // Handle subgroup scan operations.
  if (auto scanInfo = isSubgroupScan(F->getName(), F->getReturnType())) {
    const bool isInclusive = std::get<0>(*scanInfo);
    const RecurKind opKind = std::get<1>(*scanInfo);
    const bool isVP = std::get<2>(*scanInfo);
    return emitSubgroupScanBody(*F, isInclusive, opKind, isVP);
  }

  if (auto AtomicInfo = isMaskedAtomicFunction(*F)) {
    return emitMaskedAtomicBody(*F, *AtomicInfo);
  }

  return false;
}

bool VectorizationContext::emitMaskedMemOpBody(Function &F,
                                               const MemOpDesc &Desc) const {
  Value *Data = Desc.getDataOperand(&F);
  Value *Ptr = Desc.getPointerOperand(&F);
  Value *Mask = Desc.getMaskOperand(&F);
  Value *VL = Desc.isVLOp() ? Desc.getVLOperand(&F) : nullptr;
  Type *DataTy = Desc.isLoad() ? F.getReturnType() : Data->getType();

  BasicBlock *Entry = BasicBlock::Create(F.getContext(), "entry", &F);
  IRBuilder<> B(Entry);
  Value *Result = nullptr;
  if (Desc.isLoad()) {
    Result =
        VTI.createMaskedLoad(B, DataTy, Ptr, Mask, VL, Desc.getAlignment());
    B.CreateRet(Result);
  } else {
    Result = VTI.createMaskedStore(B, Data, Ptr, Mask, VL, Desc.getAlignment());
    B.CreateRetVoid();
  }
  VECZ_FAIL_IF(!Result);
  return true;
}

bool VectorizationContext::emitInterleavedMemOpBody(
    Function &F, const MemOpDesc &Desc) const {
  return emitMaskedInterleavedMemOpBody(F, Desc);
}

bool VectorizationContext::emitMaskedInterleavedMemOpBody(
    Function &F, const MemOpDesc &Desc) const {
  Value *Data = Desc.getDataOperand(&F);
  auto *const Ptr = Desc.getPointerOperand(&F);
  VECZ_FAIL_IF(!isa<VectorType>(Desc.getDataType()) || !Ptr);

  auto *const Mask = Desc.getMaskOperand(&F);
  auto *const VL = Desc.isVLOp() ? Desc.getVLOperand(&F) : nullptr;
  const auto Align = Desc.getAlignment();
  const auto Stride = Desc.getStride();

  BasicBlock *Entry = BasicBlock::Create(F.getContext(), "entry", &F);
  IRBuilder<> B(Entry);

  // If the mask is missing, assume that this is a normal interleaved memop that
  // we want to emit as an unmasked interleaved memop
  if (Desc.isLoad()) {
    auto *const Result =
        Mask ? VTI.createMaskedInterleavedLoad(B, F.getReturnType(), Ptr, Mask,
                                               Stride, VL, Align)
             : VTI.createInterleavedLoad(B, F.getReturnType(), Ptr, Stride, VL,
                                         Align);
    VECZ_FAIL_IF(!Result);
    B.CreateRet(Result);
  } else {
    auto *const Result =
        Mask ? VTI.createMaskedInterleavedStore(B, Data, Ptr, Mask, Stride, VL,
                                                Align)
             : VTI.createInterleavedStore(B, Data, Ptr, Stride, VL, Align);
    VECZ_FAIL_IF(!Result);
    B.CreateRetVoid();
  }
  return true;
}

bool VectorizationContext::emitScatterGatherMemOpBody(
    Function &F, const MemOpDesc &Desc) const {
  return emitMaskedScatterGatherMemOpBody(F, Desc);
}

bool VectorizationContext::emitMaskedScatterGatherMemOpBody(
    Function &F, const MemOpDesc &Desc) const {
  Value *Data = Desc.getDataOperand(&F);
  auto *const VecDataTy = dyn_cast<VectorType>(Desc.getDataType());
  auto *const Ptr = Desc.getPointerOperand(&F);
  VECZ_FAIL_IF(!VecDataTy || !Ptr);

  auto *const Mask = Desc.getMaskOperand(&F);
  auto *const VL = Desc.isVLOp() ? Desc.getVLOperand(&F) : nullptr;
  const auto Align = Desc.getAlignment();

  BasicBlock *Entry = BasicBlock::Create(F.getContext(), "entry", &F);
  IRBuilder<> B(Entry);

  // If the mask is missing, assume that this is a normal scatter/gather memop
  // that we want to emit as an unmasked scatter/gather memop
  if (Desc.isLoad()) {
    auto *const Result =
        Mask ? VTI.createMaskedGatherLoad(B, VecDataTy, Ptr, Mask, VL, Align)
             : VTI.createGatherLoad(B, VecDataTy, Ptr, VL, Align);
    VECZ_FAIL_IF(!Result);
    B.CreateRet(Result);
  } else {
    auto *const Result =
        Mask ? VTI.createMaskedScatterStore(B, Data, Ptr, Mask, VL, Align)
             : VTI.createScatterStore(B, Data, Ptr, VL, Align);
    VECZ_FAIL_IF(!Result);
    B.CreateRetVoid();
  }
  return true;
}

// Emit a subgroup scan operation.
// If the vectorization factor is fixed, we can do a scan in log2(N) steps,
// by noting that an inclusive scan can be split into two, and recombined into
// a single result by adding the last element of the first half onto every
// element of the second half. To deal with exclusive scans, we rotate the
// result by one element and insert the neutral element at the beginning.
//
// For now, when using scalable vectorization factor, this takes the form of a
// simple loop that accumulates the scan operation in scalar form, extracting
// and inserting elements of the resulting vector on each iteration:
//   %v = <A,B,C,D>
//   Iteration 0:
//     %e.0 = extractelement %v, 0          (A)
//     %s.0 = add N, %e.0                   (A)
//     %v.0 = insertelement undef, %s.0, 0  (<A,U,U,U>)
//   Iteration 1:
//     %e.1 = extractelement %v, 1          (B)
//     %s.1 = add %s.0, %e.1                (A+B)
//     %v.1 = insertelement  %v.0, %s.1, 1  (<A,A+B,U,U>)
//   Iteration 2:
//     %e.2 = extractelement %v, 2          (C)
//     %s.2 = add %s.1, %e.2                (A+B+C)
//     %v.2 = insertelement  %v.1, %s.2, 2  (<A,A+B,A+B+C,U>)
//   Iteration 3:
//     %e.3 = extractelement %v, 3          (D)
//     %s.3 = add %s.2, %e.2                (A+B+C+D)
//     %v.3 = insertelement  %v.2, %s.3, 3  (<A,A+B,A+B+C,A+B+C+D>)
//   Result:
//     %v.3 = <A,A+B,A+B+C,A+B+C+D>
//
// Exclusive scans operate by pre-filling the vector with the neutral value,
// looping from 1 onwards, and extracting from one less than the current
// iteration:
//   %z = insertelement undef, N, 0
//   Iteration 0:
//     %e.0 = extractelement %v, 0          (A)
//     %s.0 = add N, %e.0                   (A)
//     %v.0 = insertelement %z, %s.0, 1     (<N,A,U,U>)
// This loop operates up to the VL input, if it is a vector-predicated scan.
// Elements past the vector length will receive a default zero value.
// Note: This method is not optimal for fixed-length code, but serves as a way
// of producing scalable- and fixed-length vector code equivalently.
bool VectorizationContext::emitSubgroupScanBody(Function &F, bool IsInclusive,
                                                RecurKind OpKind,
                                                bool IsVP) const {
  LLVMContext &Ctx = F.getContext();

  auto *const Entry = BasicBlock::Create(Ctx, "entry", &F);
  IRBuilder<> B(Entry);

  Type *const VecTy = F.getReturnType();
  Type *const EltTy = multi_llvm::getVectorElementType(VecTy);
  const ElementCount EC = multi_llvm::getVectorElementCount(VecTy);

  Function::arg_iterator Arg = F.arg_begin();

  Value *const Vec = Arg;
  Value *const VL = IsVP ? ++Arg : nullptr;

  // If it's not a scalable vector, we can do it the fast way.
  if (!EC.isScalable() && !IsVP) {
    auto *const NeutralVal = compiler::utils::getNeutralVal(OpKind, EltTy);
    const auto Width = EC.getFixedValue();
    auto *const UndefVal = UndefValue::get(VecTy);

    // Put the Neutral element in a vector so we can shuffle it in.
    auto *const NeutralVec =
        B.CreateInsertElement(UndefVal, NeutralVal, B.getInt64(0));

    auto *Result = Vec;
    unsigned N = 1u;

    SmallVector<int, 16> mask(Width);
    while (N < Width) {
      // Build shuffle mask.
      // The sequence of masks will be, for a width of 16
      // (in hexadecimal for concision, where x represents the neutral value
      // element):
      //
      // x0x2x4x6x8xAxCxE
      // xx11xx55xx99xxDD
      // xxxx3333xxxxBBBB
      // xxxxxxxx77777777
      //
      const auto N2 = N << 1u;
      auto MaskIt = mask.begin();
      for (size_t i = 0; i < Width; i += N2) {
        for (size_t j = 0; j < N; ++j) {
          *MaskIt++ = Width;
        }

        const auto k = i + N - 1;
        for (size_t j = 0; j < N; ++j) {
          *MaskIt++ = k;
        }
      }
      N = N2;
      auto *const Shuffle =
          createOptimalShuffle(B, Result, NeutralVec, mask, Twine("scan_impl"));
      Result =
          compiler::utils::createBinOpForRecurKind(B, Result, Shuffle, OpKind);
    }

    if (!IsInclusive) {
      // If it is an exclusive scan, rotate the result.
      auto *const IdentityVal = compiler::utils::getIdentityVal(OpKind, EltTy);
      VECZ_FAIL_IF(!IdentityVal);
      Result = VTI.createVectorSlideUp(B, Result, IdentityVal, VL);
    }

    B.CreateRet(Result);
    return true;
  }

  // If the vector is scalable, we don't know the number of iterations required,
  // so we have to use a loop and shuffle masks generated from the step vector.

  auto *const IVTy = B.getInt32Ty();
  auto *const IndexTy = VectorType::get(IVTy, EC);
  auto *const Step = B.CreateStepVector(IndexTy, "step");
  auto *const VZero = Constant::getNullValue(IndexTy);

  auto *const Loop = BasicBlock::Create(Ctx, "loop", &F);
  auto *const Exit = BasicBlock::Create(Ctx, "exit", &F);

  // The length of the vector.
  Value *Width = nullptr;
  if (IsVP) {
    Width = VL;
  } else if (EC.isScalable()) {
    Width = B.CreateVScale(ConstantInt::get(IVTy, EC.getKnownMinValue()));
  } else {
    Width = ConstantInt::get(IVTy, EC.getFixedValue());
  }

  B.CreateBr(Loop);

  // Loop induction starts at 1 and doubles each time.
  auto *const IVStart = ConstantInt::get(IVTy, 1);

  // Create the loop instructions
  B.SetInsertPoint(Loop);

  // The induction variable (IV) which determines both our loop bounds and our
  // vector indices.
  auto *N = B.CreatePHI(IVTy, 2, "iv");
  N->addIncoming(IVStart, Entry);

  // A vector phi representing the vectorized value we're building up.
  auto *VecPhi = B.CreatePHI(VecTy, 2, "vec");
  VecPhi->addIncoming(Vec, Entry);

  // A vector phi representing the vectorized value we're building up.
  auto *MaskPhi = B.CreatePHI(IndexTy, 2, "mask.phi");
  MaskPhi->addIncoming(Step, Entry);

  // This will create shuffle masks like the following sequence:
  //
  // 1032547698BADCFE = (0123456789ABCDEF ^ splat(1))
  // 33117755BB99FFDD = (1032547698BADCFE ^ splat(2)) | splat(1)
  // 77773333FFFFBBBB = (33117755BB99FFDD ^ splat(4)) | splat(2)
  // FFFFFFFF77777777 = (77773333FFFFBBBB ^ splat(8)) | splat(4)
  //
  // We don't mix the neutral element into the vector in this case, but use a
  // Select instruction to choose between the updated or original value, so that
  // backends can lower it as a masked binary operation. The select condition
  // therefore needs to be like the following sequence:
  //
  // 0101010101010101
  // 0011001100110011
  // 0000111100001111
  // 0000000011111111

  auto *const SplatN = B.CreateVectorSplat(EC, N, "splatN");
  auto *const Mask = B.CreateXor(MaskPhi, SplatN, "mask");
  auto *const Shuffle = VTI.createVectorShuffle(B, VecPhi, Mask, VL);
  auto *const Accum =
      compiler::utils::createBinOpForRecurKind(B, VecPhi, Shuffle, OpKind);

  auto *const NBit = B.CreateAnd(MaskPhi, SplatN, "isolate");
  auto *const Which = B.CreateICmpNE(NBit, VZero, "which");
  auto *const NewVec = B.CreateSelect(Which, Accum, VecPhi, "newvec");

  auto *const NewMask = B.CreateOr(Mask, SplatN, "newmask");
  auto *const N2 = B.CreateShl(N, ConstantInt::get(IVTy, 1), "N2",
                               /*HasNUW*/ true, /*HasNSW*/ true);

  VecPhi->addIncoming(NewVec, Loop);
  MaskPhi->addIncoming(NewMask, Loop);
  N->addIncoming(N2, Loop);

  // Loop exit condition
  auto *const Cond = B.CreateICmpULT(N2, Width, "iv.cmp");
  B.CreateCondBr(Cond, Loop, Exit);

  // Function exit instructions:
  B.SetInsertPoint(Exit);

  // Create an LCSSA PHI node.
  auto *const ResultPhi = B.CreatePHI(VecTy, 1, "res.phi");
  ResultPhi->addIncoming(NewVec, Loop);

  Value *Result = ResultPhi;
  if (!IsInclusive) {
    // If it is an exclusive scan, rotate the result.
    auto *const IdentityVal = compiler::utils::getIdentityVal(OpKind, EltTy);
    VECZ_FAIL_IF(!IdentityVal);
    Result = VTI.createVectorSlideUp(B, Result, IdentityVal, VL);
  }

  B.CreateRet(Result);
  return true;
}

bool VectorizationContext::emitMaskedAtomicBody(
    Function &F, const VectorizationContext::MaskedAtomic &MA) const {
  LLVMContext &Ctx = F.getContext();
  const bool IsCmpXchg = MA.isCmpXchg();

  auto *const EntryBB = BasicBlock::Create(Ctx, "entry", &F);

  IRBuilder<> B(EntryBB);

  BasicBlock *LoopEntryBB = EntryBB;
  if (MA.IsVectorPredicated) {
    auto *const VL = F.getArg(3 + IsCmpXchg);
    // Early exit if the vector length is zero. We're going to unconditionally
    // jump into the loop after this.
    auto *const EarlyExitBB = BasicBlock::Create(Ctx, "earlyexit", &F);
    auto *const CmpZero =
        B.CreateICmpEQ(VL, ConstantInt::get(VL->getType(), 0));

    LoopEntryBB = BasicBlock::Create(Ctx, "loopentry", &F);

    B.CreateCondBr(CmpZero, EarlyExitBB, LoopEntryBB);

    B.SetInsertPoint(EarlyExitBB);
    B.CreateRet(PoisonValue::get(F.getReturnType()));
  }

  B.SetInsertPoint(LoopEntryBB);

  auto *const ExitBB = BasicBlock::Create(Ctx, "exit", &F);

  auto *const PtrArg = F.getArg(0);
  auto *const ValArg = F.getArg(1);
  Value *MaskArg = F.getArg(2 + IsCmpXchg);

  const bool IsVector = ValArg->getType()->isVectorTy();

  Value *const IdxStart = B.getInt32(0);
  ConstantInt *const KnownMin = B.getInt32(MA.VF.getKnownMinValue());
  Value *IdxEnd;
  if (MA.IsVectorPredicated) {
    IdxEnd = F.getArg(3 + IsCmpXchg);
  } else if (MA.VF.isScalable()) {
    IdxEnd = B.CreateVScale(KnownMin);
  } else {
    IdxEnd = KnownMin;
  }

  Value *RetVal = nullptr;
  Value *RetSuccessVal = nullptr;

  auto CreateLoopBody = [&MA, &F, &ExitBB, PtrArg, ValArg, MaskArg, &RetVal,
                         &RetSuccessVal, IsVector, IsCmpXchg](
                            BasicBlock *BB, Value *Idx, ArrayRef<Value *> IVs,
                            MutableArrayRef<Value *> IVsNext) -> BasicBlock * {
    IRBuilder<> IRB(BB);

    Value *MaskElt = MaskArg;
    if (IsVector) {
      MaskElt = IRB.CreateExtractElement(MaskArg, Idx, "mask");
    }
    auto *const MaskCmp =
        IRB.CreateICmpNE(MaskElt, IRB.getInt1(false), "mask.cmp");

    auto *const IfBB = BasicBlock::Create(F.getContext(), "if.then", &F);
    auto *const ElseBB = BasicBlock::Create(F.getContext(), "if.else", &F);

    IRB.CreateCondBr(MaskCmp, IfBB, ElseBB);

    {
      IRB.SetInsertPoint(IfBB);
      Value *Ptr = PtrArg;
      Value *Val = ValArg;
      if (IsVector) {
        Ptr = IRB.CreateExtractElement(PtrArg, Idx, "ptr");
        Val = IRB.CreateExtractElement(ValArg, Idx, "val");
      }

      if (IsCmpXchg) {
        Value *NewValArg = F.getArg(2);
        Value *NewVal = NewValArg;
        if (IsVector) {
          NewVal = IRB.CreateExtractElement(NewValArg, Idx, "newval");
        }
        auto *const CmpXchg =
            IRB.CreateAtomicCmpXchg(Ptr, Val, NewVal, MA.Align, MA.Ordering,
                                    *MA.CmpXchgFailureOrdering, MA.SyncScope);
        CmpXchg->setWeak(MA.IsWeak);
        CmpXchg->setVolatile(MA.IsVolatile);

        if (IsVector) {
          RetVal = IRB.CreateInsertElement(
              IVs[0], IRB.CreateExtractValue(CmpXchg, 0), Idx, "retvec");
          RetSuccessVal = IRB.CreateInsertElement(
              IVs[1], IRB.CreateExtractValue(CmpXchg, 1), Idx, "retsuccess");
        } else {
          RetVal = IRB.CreateExtractValue(CmpXchg, 0);
          RetSuccessVal = IRB.CreateExtractValue(CmpXchg, 1);
        }

      } else {
        auto *const AtomicRMW = IRB.CreateAtomicRMW(
            MA.BinOp, Ptr, Val, MA.Align, MA.Ordering, MA.SyncScope);
        AtomicRMW->setVolatile(MA.IsVolatile);

        if (IsVector) {
          RetVal = IRB.CreateInsertElement(IVs[0], AtomicRMW, Idx, "retvec");
        } else {
          RetVal = AtomicRMW;
        }
      }

      IRB.CreateBr(ElseBB);
    }

    {
      IRB.SetInsertPoint(ElseBB);

      auto *MergePhi = IRB.CreatePHI(RetVal->getType(), 2, "merge");
      MergePhi->addIncoming(IVs[0], BB);
      MergePhi->addIncoming(RetVal, IfBB);
      RetVal = MergePhi;
    }
    IVsNext[0] = RetVal;

    if (IsCmpXchg) {
      auto *MergePhi =
          IRB.CreatePHI(RetSuccessVal->getType(), 2, "mergesuccess");
      MergePhi->addIncoming(IVs[1], BB);
      MergePhi->addIncoming(RetSuccessVal, IfBB);
      RetSuccessVal = MergePhi;
      IVsNext[1] = RetSuccessVal;
    }

    // Move the exit block right to the end of the function.
    ExitBB->moveAfter(ElseBB);

    return ElseBB;
  };

  compiler::utils::CreateLoopOpts Opts;
  {
    Opts.IVs.push_back(PoisonValue::get(MA.ValTy));
    Opts.loopIVNames.push_back("retvec.prev");
  }
  if (IsCmpXchg) {
    Opts.IVs.push_back(PoisonValue::get(MaskArg->getType()));
    Opts.loopIVNames.push_back("retsuccess.prev");
  }
  compiler::utils::createLoop(LoopEntryBB, ExitBB, IdxStart, IdxEnd, Opts,
                              CreateLoopBody);

  B.SetInsertPoint(ExitBB);
  if (IsCmpXchg) {
    Value *RetStruct = PoisonValue::get(F.getReturnType());
    RetStruct = B.CreateInsertValue(RetStruct, RetVal, 0);
    RetStruct = B.CreateInsertValue(RetStruct, RetSuccessVal, 1);
    B.CreateRet(RetStruct);
  } else {
    B.CreateRet(RetVal);
  }
  return true;
}

Function *VectorizationContext::getInternalVectorEquivalent(
    Function *ScalarFn, unsigned SimdWidth) {
  // Handle masked memory loads and stores.
  if (!ScalarFn) {
    return nullptr;
  }
  if (auto Desc = MemOpDesc::analyzeMaskedMemOp(*ScalarFn)) {
    auto *NewDataTy = FixedVectorType::get(Desc->getDataType(), SimdWidth);
    return getOrCreateMaskedMemOpFn(
        *this, NewDataTy, cast<PointerType>(Desc->getPointerType()),
        Desc->getAlignment(), Desc->isLoad(), Desc->isVLOp());
  }

  return nullptr;
}

bool VectorizationContext::isMaskedFunction(const llvm::Function *F) const {
  return MaskedFunctionsMap.count(F) > 0;
}

bool VectorizationContext::insertMaskedFunction(llvm::Function *F,
                                                llvm::Function *WrappedF) {
  auto result = MaskedFunctionsMap.insert({F, WrappedF});
  return result.second;
}

llvm::Function *VectorizationContext::getOriginalMaskedFunction(
    llvm::Function *F) {
  auto Iter = MaskedFunctionsMap.find(F);
  if (Iter != MaskedFunctionsMap.end()) {
    return dyn_cast_or_null<llvm::Function>(Iter->second);
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

char DefineInternalBuiltinsPass::PassID = 0;

PreservedAnalyses DefineInternalBuiltinsPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  llvm::FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Remove internal builtins that may not be needed any more.
  SmallVector<Function *, 4> ToRemove;

  bool NonePreserved = false;
  // Implement internal builtins that we now know are needed.
  // We find all declarations that should be builtins, and then define them if
  // they have users that have associated vectorization units.
  // On failure to define, we notify those vectorization units of failure
  // and remove any partially defined body.
  // Unused declarations are removed
  for (Function &F : M.functions()) {
    if (!F.isDeclaration() || !VectorizationContext::isInternalBuiltin(&F)) {
      continue;
    }
    if (F.use_empty()) {
      ToRemove.push_back(&F);
      NonePreserved = true;
      continue;
    }
    llvm::SmallPtrSet<VectorizationUnit *, 1> UserVUs;
    for (const Use &U : F.uses()) {
      if (CallInst *CI = dyn_cast<CallInst>(U.getUser())) {
        auto R = FAM.getResult<VectorizationUnitAnalysis>(*CI->getFunction());
        if (R.hasResult()) {
          UserVUs.insert(&R.getVU());
        }
      }
    }
    if (std::all_of(UserVUs.begin(), UserVUs.end(),
                    [](VectorizationUnit *VU) { return VU->failed(); })) {
      // If the vectorization has failed, we do not want to define the internal
      // builtins, both because its a waste of time and because we might try to
      // instantiate some invalid builtin that would have been replaced by the
      // packetization process.
      continue;
    }

    VectorizationContext &Ctx = (*UserVUs.begin())->context();
    const bool DefinedBuiltin = Ctx.defineInternalBuiltin(&F);
    if (!DefinedBuiltin) {
      // If we've failed to define this builtin, ensure we clean up the
      // half-complete body. We can't simply delete it because it will have
      // uses in the vector kernel. This will revert it to a declaration, which
      // will be cleaned up later by the global optimizer.
      if (!F.isDeclaration()) {
        // defineInternalBuiltin may have partially defined the function body.
        // Clean it up. FIXME defineInternalBuiltin should probably clean up
        // after itself if there is a failure condition
        F.deleteBody();
      }
      for (VectorizationUnit *VU : UserVUs) {
        VU->setFailed("failed to define an internal builtin");
      }
      continue;
    }
    NonePreserved = true;
  }

  for (Function *F : ToRemove) {
    F->eraseFromParent();
  }

  return NonePreserved ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
