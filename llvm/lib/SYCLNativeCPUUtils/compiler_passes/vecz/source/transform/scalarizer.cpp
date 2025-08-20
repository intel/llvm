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

#include "transform/scalarizer.h"

#include <compiler/utils/builtin_info.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <multi_llvm/dibuilder.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include <algorithm>

#include "debugging.h"
#include "llvm_helpers.h"
#include "memory_operations.h"
#include "simd_packet.h"
#include "transform/printf_scalarizer.h"
#include "vectorization_context.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz-scalarization"

namespace {
/// @brief The maximum vector width that Vecz can handle.
///
/// The current limitation is due to the masks being used in the SimdPackets
/// being stored as uint64_t.
const unsigned MAX_SIMD_WIDTH = 64;
}  // namespace

using namespace vecz;
using namespace llvm;

STATISTIC(VeczScalarized, "Number of instructions scalarized [ID#S00]");
STATISTIC(VeczScalarizeFailCall,
          "Scalarize: missing function declarations [ID#S81]");
STATISTIC(VeczScalarizeFailBuiltin,
          "Scalarize: non-scalarizable builtins [ID#S82]");
STATISTIC(VeczScalarizeFailPrintf,
          "Scalarize: failures to scalarize printf [ID#S83]");
STATISTIC(VeczScalarizeFailCast,
          "Scalarize: failures to scalarize cast [ID#S84]");
STATISTIC(VeczScalarizeFailBitcast,
          "Scalarize: failures to scalarize bitcast [ID#S85]");
STATISTIC(VeczScalarizeFailReduceIntrinsic,
          "Scalarize: failures to scalarize vector.reduce intrinsic [ID#S86]");

Scalarizer::Scalarizer(llvm::Function &F, VectorizationContext &ctx,
                       bool DoubleSuport)
    : Ctx(ctx), F(F), DoubleSupport(DoubleSuport) {}

SimdPacket *Scalarizer::getPacket(const Value *V, unsigned Width, bool Create) {
  auto infoIt = packets.find(V);
  if (infoIt != packets.end()) {
    return infoIt->second.get();
  }

  if (Create) {
    auto *P = (packets[V] = std::make_unique<SimdPacket>()).get();
    P->resize(Width);
    return P;
  } else {
    return nullptr;
  }
}

Value *Scalarizer::getGather(Value *V) {
  auto &Cache = Gathers[V];
  if (Cache) {
    return Cache;
  }

  // Build the gather directly before the original instruction.
  // If it is not an instruction just return the original.
  auto *insert = dyn_cast<Instruction>(V);
  if (!insert) {
    Cache = V;
    return V;
  }

  auto *VecTy = cast<FixedVectorType>(V->getType());
  const unsigned SimdWidth = VecTy->getNumElements();

  SimdPacket *P = getPacket(V, SimdWidth, false);
  assert(P);

  // Have to build after any PHI nodes.
  while (isa<PHINode>(insert)) {
    insert = insert->getNextNode();
  }
  IRBuilder<> B(insert);

  // If every element in the packet is the same, create a vector splat instead
  // of individually inserting every element.
  Value *const splat = [](SimdPacket &P) -> Value * {
    Value *const first = P.at(0);
    for (unsigned i = 1; i < P.size(); i++) {
      if (P.at(i) != first) {
        return nullptr;
      }
    }
    return first;
  }(*P);
  if (splat) {
    return Cache =
               B.CreateVectorSplat(ElementCount::getFixed(P->size()), splat);
  }

  Value *Result = UndefValue::get(V->getType());
  for (unsigned i = 0; i < P->size(); i++) {
    if (auto *At = P->at(i)) {
      if (!isa<UndefValue>(At)) {
        Result = B.CreateInsertElement(Result, At, B.getInt32(i));
      }
    }
  }

  Cache = Result;
  return Result;
}

void Scalarizer::setNeedsScalarization(Value *V) {
  // Only mark each value once, but preserve the order
  if (ScalarizeSet.insert(V).second) {
    ToScalarize.push_back(V);
  }
}

bool Scalarizer::scalarizeAll() {
  // scalar instructions that use values to be scalarized.
  for (Value *V : ToScalarize) {
    auto *VecTy = getVectorType(V);
    assert(VecTy && "Trying to scalarize a non-vector");
    const unsigned SimdWidth = VecTy->getNumElements();
    // In the SimdPacket we use a mask that is stored as a uint64_t. Due
    // to that, there is a limit on the vector size that Vecz can
    // handle.
    VECZ_ERROR_IF(SimdWidth > MAX_SIMD_WIDTH, "The SIMD width is too large");

    PacketMask PM;
    PM.enableAll(SimdWidth);
    if (!scalarize(V, PM)) {
      return false;
    }
  }

  // Beware of instructions not being processed strictly in dominance order.
  DenseSet<Instruction *> ScalarLeaves;
  for (Value *V : ToScalarize) {
    if (Failures.contains(V)) {
      continue;
    }

    // Any user of a scalarized instruction that is not itself scalarized needs
    // its operands fixing up to use the scalarized versions.
    for (auto *U : V->users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (!ScalarizeSet.contains(I)) {
          ScalarLeaves.insert(I);
        }
      }
    }
  }

  for (Instruction *I : ScalarLeaves) {
    if (!scalarizeOperands(I)) {
      emitVeczRemarkMissed(&F, I, "Could not scalarize");
      return false;
    }
  }

  IC.deleteInstructions();
  return true;
}

Value *Scalarizer::scalarizeOperands(Instruction *I) {
  // Vector extractions.
  if (ExtractElementInst *Extract = dyn_cast<ExtractElementInst>(I)) {
    // In the SimdPacket we use a mask that is stored as a uint64_t. Due to
    // that, there is a limit on the vector size that Vecz can handle.
    VECZ_ERROR_IF(multi_llvm::getVectorNumElements(
                      Extract->getVectorOperandType()) > MAX_SIMD_WIDTH,
                  "The SIMD width is too large");
    return scalarizeOperandsExtractElement(Extract);
  }

  // Vector -> non-vector bitcasts.
  if (BitCastInst *BC = dyn_cast<BitCastInst>(I)) {
    if (BC->getSrcTy()->isVectorTy() && !BC->getDestTy()->isVectorTy()) {
      // In the SimdPacket we use a mask that is stored as a uint64_t. Due to
      // that, there is a limit on the vector size that Vecz can handle.
      VECZ_ERROR_IF(
          multi_llvm::getVectorNumElements(BC->getSrcTy()) > MAX_SIMD_WIDTH,
          "The SIMD width is too large");
      return scalarizeOperandsBitCast(BC);
    }
  }

  // printf or reduction intrinsic calls
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    Function *Callee = CI->getCalledFunction();
    VECZ_STAT_FAIL_IF(!Callee, VeczScalarizeFailCall);

    // printf calls:
    if (!Callee->isIntrinsic()) {
      // Check if this is indeed a printf call
      const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
      if (auto B = BI.analyzeBuiltin(*Callee)) {
        if (B->ID == BI.getPrintfBuiltin()) {
          return scalarizeOperandsPrintf(CI);
        }
      }
    }

    // reduction intrinsics:
    if (auto *Intrin = dyn_cast<IntrinsicInst>(CI)) {
      if (auto *reduce = scalarizeReduceIntrinsic(Intrin)) {
        return reduce;
      }
    }
  }

  // No special-case handling, so just gather any scalarized operands
  for (unsigned i = 0, n = I->getNumOperands(); i != n; ++i) {
    auto *Op = I->getOperand(i);
    if (ScalarizeSet.contains(Op)) {
      I->setOperand(i, getGather(Op));
    }
  }

  return I;
}

Value *Scalarizer::scalarizeOperandsPrintf(CallInst *CI) {
  VECZ_STAT_FAIL_IF(CI->arg_empty(), VeczScalarizeFailPrintf);

  // Get the format string as a string
  GlobalVariable *FmtStringGV = GetFormatStringAsValue(CI->getArgOperand(0));
  VECZ_STAT_FAIL_IF(!FmtStringGV, VeczScalarizeFailCall);
  const std::string FmtString = GetFormatStringAsString(FmtStringGV);
  VECZ_STAT_FAIL_IF(FmtString.empty(), VeczScalarizeFailCall);
  std::string NewFmtString;
  const EnumPrintfError err =
      ScalarizeAndCheckFormatString(FmtString, NewFmtString);
  // Check if the format string was scalarizer successfully
  VECZ_STAT_FAIL_IF(err != kPrintfError_success, VeczScalarizeFailCall);

  // Create a new global variable out of the new format string
  GlobalVariable *NewFmtStringGV = GetNewFormatStringAsGlobalVar(
      *CI->getModule(), FmtStringGV, NewFmtString);

  IRBuilder<> B(CI);
  // Gather the operands for the new printf call, taking care to scalarize
  // any vector operands.
  llvm::SmallVector<Value *, 16> NewOps;
  for (const Use &Op : CI->args()) {
    // The first operand is the new format string
    if (Op == *CI->arg_begin()) {
      Constant *Zero = B.getInt32(0);
      NewOps.push_back(B.CreateGEP(NewFmtStringGV->getValueType(),
                                   NewFmtStringGV, {Zero, Zero}));
      continue;
    }
    // The rest of the operands can either be copied or scalarized
    if (!Op->getType()->isVectorTy()) {
      // Non-vector operand, just copy
      NewOps.push_back(Op.get());
    } else {
      // Vector operand, scalarize
      // In the SimdPacket we use a mask that is stored as a uint64_t. Due
      // to that, there is a limit on the vector size that Vecz can handle.
      const uint32_t SimdWidth =
          multi_llvm::getVectorNumElements(Op->getType());
      VECZ_ERROR_IF(SimdWidth > MAX_SIMD_WIDTH, "The SIMD width is too large");
      PacketMask PM;
      PM.enableAll(SimdWidth);
      const SimdPacket *OpPacket = scalarize(Op.get(), PM);
      VECZ_STAT_FAIL_IF(!OpPacket, VeczScalarizeFailCall);
      for (unsigned i = 0; i < OpPacket->size(); ++i) {
        Value *Lane = OpPacket->at(i);
        VECZ_STAT_FAIL_IF(!Lane, VeczScalarizeFailCall);
        // We need to promote half and floats to doubles, as per 6.5.2.2/6
        // in the C99 standard, but not if the device does not have double
        // support, in which case we need to promote them to floats, as per
        // 6.12.13.2 in the OpenCL 1.2 standard.
        Type *LaneTy = Lane->getType();
        Type *PromotionType = DoubleSupport ? B.getDoubleTy() : B.getFloatTy();
        if (LaneTy->isFloatingPointTy() &&
            LaneTy->getPrimitiveSizeInBits() <
                PromotionType->getPrimitiveSizeInBits()) {
          VECZ_ERROR_IF(!LaneTy->isHalfTy() && !LaneTy->isFloatTy(),
                        "Unexpected floating point type");
          Lane = B.CreateFPExt(Lane, PromotionType);
        }
        NewOps.push_back(Lane);
      }
    }
  }
  // Create the new printf call
  Function *Callee = CI->getCalledFunction();
  CallInst *NewCI = B.CreateCall(Callee, NewOps, CI->getName());
  NewCI->setCallingConv(CI->getCallingConv());
  NewCI->setAttributes(CI->getAttributes());

  // Replace all uses of the old one with the new one
  CI->replaceAllUsesWith(NewCI);
  IC.deleteInstructionLater(CI);

  return NewCI;
}

Value *Scalarizer::scalarizeReduceIntrinsic(IntrinsicInst *Intrin) {
  // Mark unhandled reduce intrinsics to fail (for now)
  bool isHandled = true;
  Instruction::BinaryOps BinOpcode;
  switch (Intrin->getIntrinsicID()) {
    default:
      isHandled = false;
      break;
    case Intrinsic::vector_reduce_and:
      BinOpcode = Instruction::And;
      break;
    case Intrinsic::vector_reduce_or:
      BinOpcode = Instruction::Or;
      break;
    case Intrinsic::vector_reduce_xor:
      BinOpcode = Instruction::Xor;
      break;
    case Intrinsic::vector_reduce_add:
      // TODO: Need to handle FP reduce_add (Instruction::FAdd)
      if (!Intrin->getType()->isFloatTy()) {
        BinOpcode = Instruction::Add;
      } else {
        isHandled = false;
      }
      break;
    case Intrinsic::vector_reduce_mul:
      // TODO: Need to handle FP reduce_mul (Instruction::FMul)
      if (!Intrin->getType()->isFloatTy()) {
        BinOpcode = Instruction::Mul;
      } else {
        isHandled = false;
      }
      break;
    case Intrinsic::vector_reduce_fadd:
      // TODO: Need to handle FP reduce_add
      isHandled = false;
      break;
    case Intrinsic::vector_reduce_fmul:
      // TODO: Need to handle FP reduce_mul
      isHandled = false;
      break;
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_smax:
    case Intrinsic::vector_reduce_umax:
      // TODO: Need to handle Int (signed/unsigned) Max and FP Max
      isHandled = false;
      break;
    case Intrinsic::vector_reduce_fmin:
    case Intrinsic::vector_reduce_smin:
    case Intrinsic::vector_reduce_umin:
      // TODO: Need to handle Int (signed/unsigned) Min and FP Min
      isHandled = false;
      break;
  }
  // If it's an intrinsic we don't handle here, return nullptr and fallback
  // to simple gathering of any scalarized operands.
  if (!isHandled) {
    return nullptr;
  }

  // We need to handle more reduce intrinsics such as with more than 1 operand
  // like 'fadd' and 'fmul', where the first operand is scalar and the second
  // is the vector. However, the current scalarization analysis won't let these
  // through and will fail, so we the reduce intrinsic scalarization takes in
  // account only the the first (vector) operand, which is the only operand for
  // the integer reduce cases.
  Value *Vec = Intrin->getOperand(0);
  assert(Vec && "Could not get operand 0 of Intrin");

  // In the SimdPacket we use a mask that is stored as a uint64_t. Due to
  // that, there is a limit on the vector size that Vecz can handle.
  auto *VecTy = dyn_cast<FixedVectorType>(Vec->getType());
  VECZ_FAIL_IF(!VecTy);
  const uint32_t SimdWidth = VecTy->getNumElements();
  VECZ_ERROR_IF(SimdWidth > MAX_SIMD_WIDTH, "The SIMD width is too large");

  PacketMask PM;
  IRBuilder<> B(Intrin);
  PM.enableAll(SimdWidth);

  const SimdPacket *Packet = scalarize(Vec, PM);
  VECZ_STAT_FAIL_IF(!Packet, VeczScalarizeFailReduceIntrinsic);

  Type *const VecEleTy = VecTy->getElementType();
  Value *Result = ConstantInt::getNullValue(VecEleTy);
  for (unsigned i = 0; i < Packet->size(); ++i) {
    Value *const Lane = Packet->at(i);
    VECZ_STAT_FAIL_IF(!Lane, VeczScalarizeFailCall);
    Type *const LaneTy = Lane->getType();
    VECZ_ERROR_IF(LaneTy->isFloatTy(), "Unexpected floating point type");
    Result = B.CreateBinOp(BinOpcode, Result, Lane);
  }

  Intrin->replaceAllUsesWith(Result);
  IC.deleteInstructionLater(Intrin);

  return Result;
}

Value *Scalarizer::scalarizeOperandsExtractElement(ExtractElementInst *Extr) {
  // Determine the extraction index.
  Value *OrigVec = Extr->getOperand(0);
  Value *ExtractIndex = Extr->getOperand(1);
  assert(OrigVec && "Could not get operand 0 of Extr");
  assert(ExtractIndex && "Could not get operand 1 of Extr");
  ConstantInt *ConstantExtractIndex = dyn_cast<ConstantInt>(ExtractIndex);
  PacketMask PM;
  SimdPacket *OrigVecPacket;
  Value *ReturnVal;

  if (!ConstantExtractIndex) {
    // Index of extractElementInst is not a constant
    // Scalarize the original vector for all lanes.
    auto *Vec = dyn_cast<FixedVectorType>(OrigVec->getType());
    const unsigned VecWidth = Vec ? Vec->getNumElements() : 0;
    PM.enableAll(VecWidth);
    OrigVecPacket = scalarize(OrigVec, PM);
    VECZ_FAIL_IF(!OrigVecPacket);

    IRBuilder<> B(Extr);
    Value *Select = UndefValue::get(Extr->getType());
    for (unsigned lane = 0; lane < VecWidth; lane++) {
      // Check if the the lane matches the extract index and select
      // the corresponding value
      Value *Cmp = B.CreateICmpEQ(
          ConstantInt::get(ExtractIndex->getType(), lane), ExtractIndex);
      Select = B.CreateSelect(Cmp, OrigVecPacket->at(lane), Select);
    }
    ReturnVal = Select;
  } else {
    // Scalarize the original vector, but only for the lane to extract.
    const unsigned Lane = ConstantExtractIndex->getZExtValue();
    PM.enable(Lane);
    OrigVecPacket = scalarize(OrigVec, PM);
    VECZ_FAIL_IF(!OrigVecPacket);
    ReturnVal = OrigVecPacket->at(Lane);
  }

  // Replace the extraction by the extracted lane value.
  Extr->replaceAllUsesWith(ReturnVal);
  IC.deleteInstructionLater(Extr);
  return ReturnVal;
}

Value *Scalarizer::scalarizeOperandsBitCast(BitCastInst *BC) {
  auto *VecSrcTy = dyn_cast<FixedVectorType>(BC->getSrcTy());
  VECZ_FAIL_IF(!VecSrcTy);
  const unsigned SimdWidth = VecSrcTy->getNumElements();
  PacketMask PM;
  PM.enableAll(SimdWidth);
  const SimdPacket *SrcPacket = scalarize(BC->getOperand(0), PM);
  VECZ_FAIL_IF(!SrcPacket);

  Type *DstTy = BC->getDestTy();
  Type *DstAsIntTy = DstTy;
  Type *SrcEleTy = VecSrcTy->getElementType();
  Type *SrcEleAsIntTy = SrcEleTy;
  const uint64_t SrcEleBits = SrcEleTy->getScalarSizeInBits();
  const uint64_t DstBits = DstTy->getPrimitiveSizeInBits();
  if (!DstTy->isIntegerTy()) {
    DstAsIntTy = IntegerType::get(BC->getContext(), DstBits);
  }
  if (!SrcEleTy->isIntegerTy()) {
    SrcEleAsIntTy = IntegerType::get(BC->getContext(), SrcEleBits);
  }

  // Successively OR each scalarized value together.
  IRBuilder<> B(BC);
  Value *Result = ConstantInt::getNullValue(DstAsIntTy);
  for (unsigned i = 0; i < SimdWidth; i++) {
    Value *Lane = SrcPacket->at(i);
    if (!SrcEleTy->isIntegerTy()) {
      Lane = B.CreateBitCast(Lane, SrcEleAsIntTy);
    }
    Lane = B.CreateZExt(Lane, DstAsIntTy);
    Lane = B.CreateShl(Lane, i * SrcEleBits);
    Result = B.CreateOr(Result, Lane);
  }
  if (!DstTy->isIntegerTy()) {
    Result = B.CreateBitCast(Result, DstTy);
  }
  BC->replaceAllUsesWith(Result);
  IC.deleteInstructionLater(BC);
  return Result;
}

SimdPacket *Scalarizer::scalarize(Value *V, PacketMask PM) {
  auto *VecTy = getVectorType(V);
  VECZ_ERROR_IF(!VecTy,
                "We shouldn't be trying to scalarize a non-vector instruction");
  const unsigned SimdWidth = VecTy->getNumElements();

  // Re-use cached packets, but make sure it contains all the lanes we want.
  // If we have a cached packet with missing lanes, it will be fetched by
  // getPacket and filled with the new lanes.
  SimdPacket *CachedPacket = getPacket(V, SimdWidth, false);
  if (CachedPacket && ((CachedPacket->Mask.Value & PM.Value) == PM.Value)) {
    return CachedPacket;
  }

  // This value hasn't been scheduled for scalarization, so extract instead
  if (!V->getType()->isVoidTy() && !ScalarizeSet.contains(V)) {
    return extractLanes(V, PM);
  }

  // Only instructions can be scalarized at this point.
  Instruction *Ins = dyn_cast<Instruction>(V);
  if (!Ins) {
    if (!V->getType()->isVoidTy()) {
      return extractLanes(V, PM);
    } else {
      return assignScalar(nullptr, V);
    }
  }

  // Figure out what kind of instruction it is and try to scalarize it.
  SimdPacket *Result = nullptr;
  switch (Ins->getOpcode()) {
    default:
      if (Ins->isBinaryOp()) {
        Result = scalarizeBinaryOp(cast<BinaryOperator>(V), PM);
      } else if (Ins->isCast()) {
        Result = scalarizeCast(cast<CastInst>(V), PM);
      } else if (Ins->isUnaryOp()) {
        Result = scalarizeUnaryOp(cast<UnaryOperator>(V), PM);
      }
      break;
    case Instruction::GetElementPtr:
      Result = scalarizeGEP(cast<GetElementPtrInst>(V), PM);
      break;
    case Instruction::Store:
      Result = scalarizeStore(cast<StoreInst>(V), PM);
      break;
    case Instruction::Load:
      Result = scalarizeLoad(cast<LoadInst>(V), PM);
      break;
    case Instruction::Call:
      Result = scalarizeCall(cast<CallInst>(V), PM);
      break;
    case Instruction::ICmp:
      Result = scalarizeICmp(cast<ICmpInst>(V), PM);
      break;
    case Instruction::FCmp:
      Result = scalarizeFCmp(cast<FCmpInst>(V), PM);
      break;
    case Instruction::Select:
      Result = scalarizeSelect(cast<SelectInst>(V), PM);
      break;
    case Instruction::ShuffleVector:
      Result = scalarizeShuffleVector(cast<ShuffleVectorInst>(V), PM);
      break;
    case Instruction::InsertElement:
      Result = scalarizeInsertElement(cast<InsertElementInst>(V), PM);
      break;
    case Instruction::PHI:
      Result = scalarizePHI(cast<PHINode>(V), PM);
      break;
      // Freeze instruction is not available in LLVM versions prior 10.0
      // and not used in LLVM versions prior to 11.0
    case Instruction::Freeze:
      Result = scalarizeFreeze(cast<FreezeInst>(V), PM);
      break;
  }

  if (Result) {
    scalarizeDI(Ins, Result, SimdWidth);
    return assignScalar(Result, V);
  } else {
    // If an instruction couldn't be scalarized, we can just extract its
    // elements, but we also need to remove it from the scalarization set and
    // add it to the failures set so any scalar leaves don't try to scalarize
    // it again.
    ScalarizeSet.erase(Ins);
    Failures.insert(Ins);
    return extractLanes(V, PM);
  }
}

SimdPacket *Scalarizer::extractLanes(llvm::Value *V, PacketMask PM) {
  auto *VecTy = getVectorType(V);
  VECZ_FAIL_IF(!VecTy);
  const unsigned SimdWidth = VecTy->getNumElements();
  SimdPacket *P = getPacket(V, SimdWidth);

  if (Constant *CVec = dyn_cast<Constant>(V)) {
    assert(isa<FixedVectorType>(CVec->getType()) && "Invalid constant type!");
    SimdPacket *P = getPacket(CVec, SimdWidth);
    for (unsigned i = 0; i < SimdWidth; i++) {
      if (!PM.isEnabled(i) || P->at(i)) {
        continue;
      }
      P->set(i, CVec->getAggregateElement(i));
    }
    return P;
  }

  Instruction *insert = nullptr;

  if (auto *Arg = dyn_cast<Argument>(V)) {
    BasicBlock &Entry = Arg->getParent()->getEntryBlock();

    // Make sure we start inserting new instructions after any allocas
    auto insertAfter = Entry.begin();

    while (isa<AllocaInst>(*insertAfter)) {
      insertAfter++;
    }
    insert = &*insertAfter;
  } else if (auto *Inst = dyn_cast<Instruction>(V)) {
    insert = Inst->getNextNode();
    while (isa<PHINode>(insert)) {
      insert = insert->getNextNode();
    }
  } else {
    return nullptr;
  }

  const SimplifyQuery Q(F.getParent()->getDataLayout());

  IRBuilder<> B(insert);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }

    Value *Idx = B.getInt32(i);
    Value *Extract = simplifyExtractElementInst(V, Idx, Q);
    if (!Extract) {
      Extract = B.CreateExtractElement(V, Idx);
    }
    P->set(i, Extract);
  }
  return P;
}

void Scalarizer::scalarizeDI(Instruction *Original, const SimdPacket *Packet,
                             unsigned Width) {
  // Don't support scalarizing PHI nodes
  if (!Packet || !Original || isa<PHINode>(Original)) {
    return;
  }

  auto *const LAM = LocalAsMetadata::getIfExists(Original);
  if (!LAM) {
    return;
  }

  // Contains processed SIMD values for which we create scalar debug
  // instructions and is used to avoid duplicate LLVM dbg.value's.
  SmallPtrSet<Value *, 4> VectorElements;

  multi_llvm::DIBuilder DIB(*Original->getModule(), false);

  for (DbgVariableRecord *const DVR : LAM->getAllDbgVariableRecordUsers()) {
    DILocalVariable *DILocal = nullptr;
    DebugLoc DILoc;

    switch (DVR->getType()) {
      case DbgVariableRecord::LocationType::Value:
      case DbgVariableRecord::LocationType::Declare:
        DILocal = DVR->getVariable();
        DILoc = DVR->getDebugLoc();
        break;
      default:
        continue;
    }

    // Create new DbgVariableRecord across enabled SIMD lanes
    const auto bitSize = Original->getType()->getScalarSizeInBits();
    for (unsigned lane = 0; lane < Width; ++lane) {
      Value *LaneVal = Packet->at(lane);
      if (LaneVal && !isa<UndefValue>(LaneVal)) {
        // Check if the LaneVal SIMD Value is already processed
        // and a Debug Value Intrinsic has been created for it.
        if (VectorElements.contains(LaneVal)) {
          continue;
        }
        // DWARF bit piece expressions are used to describe part of an
        // aggregate variable, our vector, which is fragmented across multiple
        // values. First argument takes the offset of the piece, and the second
        // takes the piece size.
        std::optional<DIExpression *> DIExpr =
            DIExpression::createFragmentExpression(DIB.createExpression(),
                                                   lane * bitSize, bitSize);
        if (DIExpr) {
          DIB.insertDbgValueIntrinsic(LaneVal, DILocal, *DIExpr, DILoc,
                                      Original->getIterator());
          VectorElements.insert(LaneVal);
        }
      }
    }
  }

  auto *const MDV = MetadataAsValue::getIfExists(Original->getContext(), LAM);
  if (!MDV) {
    return;
  }
}

SimdPacket *Scalarizer::assignScalar(SimdPacket *P, Value *V) {
  if (!P) {
    emitVeczRemarkMissed(&F, V, "Could not scalarize");
  } else {
    ++VeczScalarized;
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      IC.deleteInstructionLater(I);
    }
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeLoad(LoadInst *Load, PacketMask PM) {
  Value *PtrBase = Load->getPointerOperand();
  auto *VecDataTy = dyn_cast<FixedVectorType>(Load->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();

  Type *ScalarEleTy = VecDataTy->getElementType();

  // Absorb redundant bitcasts
  GetElementPtrInst *PtrGEP = dyn_cast<GetElementPtrInst>(PtrBase);
  const bool InBounds = (PtrGEP && PtrGEP->isInBounds());

  IRBuilder<> B(Load);

  SimdPacket PtrPacket;
  SimdPacket *P = getPacket(Load, SimdWidth);
  PtrPacket.resize(SimdWidth);

  // Emit scalarized pointers.
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || PtrPacket.at(i)) {
      continue;
    }

    // Re-use GEPs if available
    if (P->at(i)) {
      LoadInst *LoadI = cast<LoadInst>(P->at(i));
      Value *PtrI = LoadI->getPointerOperand();
      if (isa<GetElementPtrInst>(PtrI)) {
        PtrPacket.set(i, PtrI);
        continue;
      }
    }

    Value *Ptr = InBounds
                     ? B.CreateInBoundsGEP(ScalarEleTy, PtrBase, B.getInt32(i))
                     : B.CreateGEP(ScalarEleTy, PtrBase, B.getInt32(i));
    PtrPacket.set(i, Ptr);
  }

  // The individual elements may need laxer alignment requirements than the
  // whole vector.
  const unsigned Alignment = Load->getAlign().value();
  unsigned EleAlign = ScalarEleTy->getPrimitiveSizeInBits() / 8;
  EleAlign = std::min(Alignment, EleAlign);

  // Emit scalarized loads.
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    LoadInst *NewLoad = B.CreateLoad(ScalarEleTy, PtrPacket.at(i),
                                     Load->isVolatile(), Load->getName());

    NewLoad->copyMetadata(*Load);
    NewLoad->setAlignment(MaybeAlign(EleAlign).valueOrOne());

    P->set(i, NewLoad);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeStore(StoreInst *Store, PacketMask PM) {
  Value *PtrBase = Store->getPointerOperand();
  assert(PtrBase && "Could not get pointer operand from Store");
  auto *VecDataTy =
      dyn_cast<FixedVectorType>(Store->getValueOperand()->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  Type *ScalarEleTy = VecDataTy->getElementType();
  Value *VectorData = Store->getValueOperand();

  // Emit scalarized data values.
  const SimdPacket *DataPacket = scalarize(VectorData, PM);
  VECZ_FAIL_IF(!DataPacket);

  GetElementPtrInst *PtrGEP = dyn_cast<GetElementPtrInst>(PtrBase);
  const bool InBounds = (PtrGEP && PtrGEP->isInBounds());

  IRBuilder<> B(Store);

  SimdPacket PtrPacket;
  SimdPacket *P = getPacket(Store, SimdWidth);
  PtrPacket.resize(SimdWidth);

  // Emit scalarized pointers.
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || PtrPacket.at(i)) {
      continue;
    }

    // Re-use GEPs if available
    if (P->at(i)) {
      StoreInst *StoreI = cast<StoreInst>(P->at(i));
      Value *PtrI = StoreI->getPointerOperand();
      if (isa<GetElementPtrInst>(PtrI)) {
        PtrPacket.set(i, PtrI);
        continue;
      }
    }

    Value *Ptr = InBounds
                     ? B.CreateInBoundsGEP(ScalarEleTy, PtrBase, B.getInt32(i))
                     : B.CreateGEP(ScalarEleTy, PtrBase, B.getInt32(i));
    PtrPacket.set(i, Ptr);
  }

  // See comment at equivalent part of scalarizeLoad()
  const unsigned Alignment = Store->getAlign().value();
  unsigned EleAlign = ScalarEleTy->getPrimitiveSizeInBits() / 8;
  EleAlign = std::min(Alignment, EleAlign);

  // Emit scalarized stores.
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *Data = DataPacket->at(i);
    if (isa<UndefValue>(Data)) {
      P->set(i, Data);
    } else {
      StoreInst *NewStore =
          B.CreateStore(Data, PtrPacket.at(i), Store->isVolatile());

      NewStore->copyMetadata(*Store);
      NewStore->setAlignment(MaybeAlign(EleAlign).valueOrOne());

      P->set(i, NewStore);
    }
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeBinaryOp(BinaryOperator *BinOp,
                                          PacketMask PM) {
  IRBuilder<> B(BinOp);
  Value *LHS = BinOp->getOperand(0);
  auto *VecDataTy = dyn_cast<FixedVectorType>(LHS->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  const SimdPacket *LHSPacket = scalarize(LHS, PM);
  VECZ_FAIL_IF(!LHSPacket);
  Value *RHS = BinOp->getOperand(1);
  const SimdPacket *RHSPacket = scalarize(RHS, PM);
  VECZ_FAIL_IF(!RHSPacket);
  SimdPacket *P = getPacket(BinOp, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *New = B.CreateBinOp(BinOp->getOpcode(), LHSPacket->at(i),
                               RHSPacket->at(i), BinOp->getName());
    if (BinaryOperator *NewBinOp = dyn_cast<BinaryOperator>(New)) {
      NewBinOp->copyIRFlags(BinOp);
    }
    P->set(i, New);
  }
  return P;
}

// Freeze instruction is not available in LLVM versions prior 10.0
// and not used in LLVM versions prior to 11.0
SimdPacket *Scalarizer::scalarizeFreeze(FreezeInst *FreezeI, PacketMask PM) {
  IRBuilder<> B(FreezeI);
  Value *Src = FreezeI->getOperand(0);
  auto *VecDataTy = dyn_cast<FixedVectorType>(Src->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  const SimdPacket *SrcPacket = scalarize(Src, PM);
  VECZ_FAIL_IF(!SrcPacket);

  // Create scalarized freeze.
  SimdPacket *P = getPacket(FreezeI, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *New = B.CreateFreeze(SrcPacket->at(i), FreezeI->getName());
    P->set(i, New);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeUnaryOp(UnaryOperator *UnOp, PacketMask PM) {
  IRBuilder<> B(UnOp);
  Value *Src = UnOp->getOperand(0);
  auto *VecDataTy = dyn_cast<FixedVectorType>(Src->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  const SimdPacket *SrcPacket = scalarize(Src, PM);
  VECZ_FAIL_IF(!SrcPacket);
  SimdPacket *P = getPacket(UnOp, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *New =
        B.CreateUnOp(UnOp->getOpcode(), SrcPacket->at(i), UnOp->getName());
    if (UnaryOperator *NewUnOp = dyn_cast<UnaryOperator>(New)) {
      NewUnOp->copyIRFlags(UnOp);
    }
    P->set(i, New);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeCast(CastInst *CastI, PacketMask PM) {
  // Make sure we support the cast operation.
  const CastInst::CastOps Opc = CastI->getOpcode();
  switch (Opc) {
    default:
      return nullptr;
    case CastInst::BitCast:
      return scalarizeBitCast(cast<BitCastInst>(CastI), PM);
    case CastInst::Trunc:
    case CastInst::ZExt:
    case CastInst::SExt:
    case CastInst::FPToUI:
    case CastInst::FPToSI:
    case CastInst::UIToFP:
    case CastInst::SIToFP:
    case CastInst::FPTrunc:
    case CastInst::FPExt:
    case CastInst::AddrSpaceCast:
      break;
  }

  // Scalarize the source value.
  IRBuilder<> B(CastI);
  Value *Src = CastI->getOperand(0);
  auto *VecSrcTy = dyn_cast<FixedVectorType>(Src->getType());
  VECZ_FAIL_IF(!VecSrcTy);
  const unsigned SimdWidth = VecSrcTy->getNumElements();
  auto *VecDstTy = dyn_cast<FixedVectorType>(CastI->getType());
  VECZ_STAT_FAIL_IF(!VecDstTy || (VecDstTy->getNumElements() != SimdWidth),
                    VeczScalarizeFailCast);
  const SimdPacket *SrcPacket = scalarize(Src, PM);
  VECZ_FAIL_IF(!SrcPacket);

  // Create scalarized casts.
  SimdPacket *P = getPacket(CastI, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    auto *const SrcPacketValue = SrcPacket->at(i);
    VECZ_FAIL_IF(!SrcPacketValue);
    Value *New = B.CreateCast(Opc, SrcPacketValue, VecDstTy->getElementType(),
                              CastI->getName());
    P->set(i, New);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeBitCast(BitCastInst *BC, PacketMask PM) {
  IRBuilder<> B(BC);
  Type *SrcTy = BC->getSrcTy();
  Value *Src = BC->getOperand(0);
  auto *VecSrcTy = dyn_cast<FixedVectorType>(SrcTy);
  auto *VecDstTy = dyn_cast<FixedVectorType>(BC->getDestTy());
  VECZ_FAIL_IF(!VecDstTy);
  const unsigned SimdWidth = VecDstTy->getNumElements();
  const bool Vec3Src = VecSrcTy && (VecSrcTy->getNumElements() == 3);
  const bool Vec3Dst = (SimdWidth == 3);
  VECZ_STAT_FAIL_IF(Vec3Src ^ Vec3Dst, VeczScalarizeFailBitcast);

  // Handle non-vector -> vector casts and vector casts with different widths.
  if (!VecSrcTy || (VecSrcTy->getNumElements() != SimdWidth)) {
    VECZ_FAIL_IF(BC->getModule()->getDataLayout().isBigEndian());

    // Treat scalars as vectors of length 1.
    SimdPacket SrcScalar{Src};
    SimdPacket &S =
        VecSrcTy ? *getPacket(Src, VecSrcTy->getNumElements()) : SrcScalar;
    Type *const SrcEleTy = VecSrcTy ? VecSrcTy->getElementType() : SrcTy;
    // Source element need not be a primitive if it was a non-vector, but in
    // that case we know the size must match the destination vector type.
    const size_t SrcEleSize = VecSrcTy ? SrcEleTy->getPrimitiveSizeInBits()
                                       : VecDstTy->getPrimitiveSizeInBits();
    Type *const SrcEleIntTy =
        SrcEleTy->isIntegerTy()
            ? SrcEleTy
            : SrcEleTy->getIntNTy(BC->getContext(),
                                  SrcEleTy->getPrimitiveSizeInBits());
    Type *const DstEleTy = VecDstTy->getElementType();
    const size_t DstEleSize = DstEleTy->getPrimitiveSizeInBits();
    Type *const DstEleIntTy =
        DstEleTy->isIntegerTy()
            ? DstEleTy
            : DstEleTy->getIntNTy(BC->getContext(),
                                  DstEleTy->getPrimitiveSizeInBits());
    SimdPacket *P = getPacket(BC, SimdWidth);
    PacketMask SPM;
    for (unsigned i = 0; i < SimdWidth; i++) {
      if (!PM.isEnabled(i) || P->at(i)) {
        continue;
      }
      if (VecSrcTy) {
        for (unsigned j = i * DstEleSize / SrcEleSize;
             j * SrcEleSize < (i + 1) * DstEleSize; ++j) {
          SPM.enable(j);
        }
        const SimdPacket *SrcPacket = scalarize(Src, SPM);
        VECZ_FAIL_IF(!SrcPacket);
        assert(SrcPacket == &S &&
               "Scalarization of Src should update existing packet");
      }
      Value *Lane = nullptr;
      for (unsigned j = i * DstEleSize / SrcEleSize;
           j * SrcEleSize < (i + 1) * DstEleSize; ++j) {
        Value *SrcPart = S[j];
        assert(
            SrcPart &&
            "Scalarization of Src failure should have been detected earlier");
        if (SrcEleIntTy != SrcEleTy) {
          SrcPart = B.CreateBitCast(SrcPart, SrcEleIntTy);
        }
        if (SrcEleIntTy->getIntegerBitWidth() <
            DstEleIntTy->getIntegerBitWidth()) {
          SrcPart = B.CreateZExt(SrcPart, DstEleIntTy);
        }
        if (i * DstEleSize > j * SrcEleSize) {
          SrcPart = B.CreateLShr(SrcPart, (i * DstEleSize) - (j * SrcEleSize));
        } else if (j * SrcEleSize > i * DstEleSize) {
          SrcPart = B.CreateShl(SrcPart, (j * SrcEleSize) - (i * DstEleSize));
        }
        if (SrcEleIntTy->getIntegerBitWidth() >
            DstEleIntTy->getIntegerBitWidth()) {
          SrcPart = B.CreateTrunc(SrcPart, DstEleIntTy);
        }
        Lane = Lane ? B.CreateOr(Lane, SrcPart) : SrcPart;
      }
      assert(Lane && "No bits found for lane");
      if (DstEleTy != DstEleIntTy) {
        Lane = B.CreateBitCast(Lane, DstEleTy);
      }
      P->set(i, Lane);
    }
    return P;
  }

  // Handle same width vector -> vector casts, quite a more straighforward
  // affair.
  const SimdPacket *SrcPacket = scalarize(Src, PM);
  VECZ_FAIL_IF(!SrcPacket);
  Type *DstEleTy = VecDstTy->getElementType();
  SimdPacket *P = getPacket(BC, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *NewVal = B.CreateBitCast(SrcPacket->at(i), DstEleTy);
    P->set(i, NewVal);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeICmp(ICmpInst *ICmp, PacketMask PM) {
  IRBuilder<> B(ICmp);
  Value *LHS = ICmp->getOperand(0);
  auto *VecDataTy = dyn_cast<FixedVectorType>(ICmp->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  const SimdPacket *LHSPacket = scalarize(LHS, PM);
  VECZ_FAIL_IF(!LHSPacket);
  Value *RHS = ICmp->getOperand(1);
  const SimdPacket *RHSPacket = scalarize(RHS, PM);
  VECZ_FAIL_IF(!RHSPacket);
  SimdPacket *P = getPacket(ICmp, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *New = B.CreateICmp(ICmp->getPredicate(), LHSPacket->at(i),
                              RHSPacket->at(i), ICmp->getName());
    P->set(i, New);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeFCmp(FCmpInst *FCmp, PacketMask PM) {
  IRBuilder<> B(FCmp);
  Value *LHS = FCmp->getOperand(0);
  auto *VecDataTy = dyn_cast<FixedVectorType>(FCmp->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  const SimdPacket *LHSPacket = scalarize(LHS, PM);
  VECZ_FAIL_IF(!LHSPacket);
  Value *RHS = FCmp->getOperand(1);
  const SimdPacket *RHSPacket = scalarize(RHS, PM);
  VECZ_FAIL_IF(!RHSPacket);
  SimdPacket *P = getPacket(FCmp, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *New = B.CreateFCmp(FCmp->getPredicate(), LHSPacket->at(i),
                              RHSPacket->at(i), FCmp->getName());
    P->set(i, New);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeSelect(SelectInst *Select, PacketMask PM) {
  IRBuilder<> B(Select);
  Value *Cond = Select->getCondition();
  const SimdPacket *CondPacket = nullptr;
  if (Cond->getType()->isVectorTy()) {
    CondPacket = scalarize(Cond, PM);
    VECZ_FAIL_IF(!CondPacket);
  }
  Value *TrueVal = Select->getTrueValue();
  auto *VecDataTy = dyn_cast<FixedVectorType>(Select->getType());
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  const SimdPacket *TruePacket = scalarize(TrueVal, PM);
  VECZ_FAIL_IF(!TruePacket);
  Value *FalseVal = Select->getFalseValue();
  const SimdPacket *FalsePacket = scalarize(FalseVal, PM);
  VECZ_FAIL_IF(!FalsePacket);
  SimdPacket *P = getPacket(Select, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *CondLane = CondPacket ? CondPacket->at(i) : Cond;
    Value *New = B.CreateSelect(CondLane, TruePacket->at(i), FalsePacket->at(i),
                                Select->getName());
    P->set(i, New);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeMaskedMemOp(CallInst *CI, PacketMask PM,
                                             MemOp &MaskedOp) {
  Function *Callee = CI->getCalledFunction();
  VECZ_STAT_FAIL_IF(!Callee, VeczScalarizeFailCall);
  auto *VecDataTy = getVectorType(CI);
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();
  assert((MaskedOp.isLoad() || MaskedOp.isStore()) &&
         "Masked op is not a store or load!");

  // Scalarize mask
  Value *MaskOperand = MaskedOp.getMaskOperand();
  VECZ_FAIL_IF(!MaskOperand);
  const SimdPacket *MaskPacket = scalarize(MaskedOp.getMaskOperand(), PM);
  VECZ_FAIL_IF(!MaskPacket);

  Value *PtrBase = MaskedOp.getPointerOperand();
  VECZ_FAIL_IF(!PtrBase);

  // Scalarize data packet if this is a store
  const SimdPacket *DataPacket = nullptr;
  if (MaskedOp.isStore()) {
    DataPacket = scalarize(MaskedOp.getDataOperand(), PM);
    VECZ_FAIL_IF(!DataPacket);
  }

  Type *ScalarEleTy = VecDataTy->getElementType();

  GetElementPtrInst *PtrGEP = dyn_cast<GetElementPtrInst>(PtrBase);
  const bool InBounds = (PtrGEP && PtrGEP->isInBounds());

  IRBuilder<> B(CI);

  SimdPacket PtrPacket;
  SimdPacket *P = getPacket(CI, SimdWidth);
  PtrPacket.resize(SimdWidth);

  // Create scalar pointers
  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || PtrPacket.at(i)) {
      continue;
    }

    Value *Ptr = InBounds
                     ? B.CreateInBoundsGEP(ScalarEleTy, PtrBase, B.getInt32(i))
                     : B.CreateGEP(ScalarEleTy, PtrBase, B.getInt32(i));
    PtrPacket.set(i, Ptr);
  }

  const unsigned Alignment = MaskedOp.getAlignment();
  unsigned EleAlign = ScalarEleTy->getPrimitiveSizeInBits() / 8;
  EleAlign = std::min(Alignment, EleAlign);

  for (unsigned i = 0; i < SimdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Instruction *ScalarMemOp = nullptr;
    if (MaskedOp.isLoad()) {
      ScalarMemOp =
          createMaskedLoad(Ctx, ScalarEleTy, PtrPacket.at(i), MaskPacket->at(i),
                           /*EVL*/ nullptr, EleAlign);
    } else {
      ScalarMemOp = createMaskedStore(Ctx, DataPacket->at(i), PtrPacket.at(i),
                                      MaskPacket->at(i),
                                      /*EVL*/ nullptr, EleAlign);
    }
    VECZ_FAIL_IF(!ScalarMemOp);
    B.Insert(ScalarMemOp);
    P->set(i, ScalarMemOp);
  }

  return P;
}

SimdPacket *Scalarizer::scalarizeCall(CallInst *CI, PacketMask PM) {
  compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  Function *Callee = CI->getCalledFunction();
  VECZ_STAT_FAIL_IF(!Callee, VeczScalarizeFailCall);
  auto *VecDataTy = getVectorType(CI);
  VECZ_FAIL_IF(!VecDataTy);
  const unsigned SimdWidth = VecDataTy->getNumElements();

  if (auto MaskedOp = MemOp::get(CI, MemOpAccessKind::Masked)) {
    if (MaskedOp->isMaskedMemOp()) {
      return scalarizeMaskedMemOp(CI, PM, *MaskedOp);
    }
  }

  Value *VectorCallMask = nullptr;
  if (Ctx.isMaskedFunction(Callee)) {
    // We have a masked call to a function.
    // Extract the mask from the call, we need to re-apply it later
    VectorCallMask = CI->getArgOperand(CI->arg_size() - 1);

    // Get the original function call from the masked wrapper function
    Function *originalFunc = Ctx.getOriginalMaskedFunction(Callee);
    Callee = originalFunc;
  }

  const auto Builtin = BI.analyzeBuiltin(*Callee);
  VECZ_FAIL_IF(!Builtin);
  Function *ScalarEquiv = BI.getScalarEquivalent(*Builtin, F.getParent());
  VECZ_STAT_FAIL_IF(!ScalarEquiv, VeczScalarizeFailBuiltin);

  IRBuilder<> B(CI);
  const auto Props = Builtin->properties;
  // Ignore the mask if present
  const unsigned NumArgs = VectorCallMask ? CI->arg_size() - 1 : CI->arg_size();
  SmallVector<SimdPacket *, 4> OpPackets(NumArgs);
  SmallVector<Value *, 4> OpScalars(NumArgs);
  for (unsigned i = 0; i < NumArgs; i++) {
    Value *OrigOp = CI->getArgOperand(i);
    Type *OldTy = OrigOp->getType();
    if (OldTy->isVectorTy()) {
      SimdPacket *OpPacket = scalarize(OrigOp, PM);
      VECZ_FAIL_IF(!OpPacket);
      OpPackets[i] = OpPacket;
    } else if (PointerType *OldPtrTy = dyn_cast<PointerType>(OldTy)) {
      auto *const PtrRetPointeeTy =
          compiler::utils::getPointerReturnPointeeTy(*Callee, Props);
      if (PtrRetPointeeTy && PtrRetPointeeTy->isVectorTy()) {
        // Handle 'pointer return' arguments. The old type was Vector*, the new
        // type is Scalar*. To accommodate the different we need to have
        // individual offsets, one for each 'element pointer'.
        auto *OldVecTy = cast<FixedVectorType>(PtrRetPointeeTy);
        VECZ_STAT_FAIL_IF(OldVecTy->getNumElements() != SimdWidth,
                          VeczScalarizeFailBuiltin);
        Type *NewTy = OldPtrTy;
        Value *ScalarAddrBase = B.CreateBitCast(OrigOp, NewTy);
        SimdPacket *OpPacket = getPacket(ScalarAddrBase, SimdWidth);
        for (unsigned j = 0; j < SimdWidth; j++) {
          if (!PM.isEnabled(j) || OpPacket->at(j)) {
            continue;
          }
          Value *ScalarAddr = B.CreateGEP(OldVecTy->getElementType(),
                                          ScalarAddrBase, B.getInt32(j));
          OpPacket->set(j, ScalarAddr);
          OpPackets[i] = OpPacket;
        }
      } else {
        OpScalars[i] = OrigOp;
      }
    } else {
      OpScalars[i] = OrigOp;
    }
  }

  SimdPacket *P = getPacket(CI, SimdWidth);
  for (unsigned j = 0; j < SimdWidth; j++) {
    if (!PM.isEnabled(j) || P->at(j)) {
      continue;
    }
    SmallVector<Value *, 4> Ops;
    for (unsigned i = 0; i < NumArgs; i++) {
      const SimdPacket *OpPacket = OpPackets[i];
      if (OpPacket) {
        Ops.push_back(OpPacket->at(j));
      } else {
        Value *OrigOp = OpScalars[i];
        VECZ_FAIL_IF(!OrigOp);
        Ops.push_back(OrigOp);
      }
    }

    CallInst *NewCI = B.CreateCall(ScalarEquiv, Ops, CI->getName());
    NewCI->setCallingConv(CI->getCallingConv());
    NewCI->setAttributes(CI->getAttributes());
    // Re-apply mask. The new CI already has to exist to create the masked
    // function which is why it gets updated here. We then need to add the
    // mask argument back to the call, but LLVM won't let us update the existing
    // one, so recreate the CallInst one last time
    if (VectorCallMask) {
      Function *MaskedScalarEquiv = Ctx.getOrCreateMaskedFunction(NewCI);
      VECZ_FAIL_IF(!MaskedScalarEquiv);
      Ops.push_back(VectorCallMask);
      CallInst *NewCIMasked =
          B.CreateCall(MaskedScalarEquiv, Ops, CI->getName());
      NewCIMasked->setCallingConv(CI->getCallingConv());
      NewCIMasked->setAttributes(CI->getAttributes());
      P->set(j, NewCIMasked);
      NewCI->eraseFromParent();
    } else {
      P->set(j, NewCI);
    }
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeShuffleVector(ShuffleVectorInst *Shuffle,
                                               PacketMask PM) {
  auto *VecTy = dyn_cast<FixedVectorType>(Shuffle->getType());
  VECZ_FAIL_IF(!VecTy);
  Value *LHS = Shuffle->getOperand(0);
  Value *RHS = Shuffle->getOperand(1);
  assert(LHS && "Could not get operand 0");
  assert(RHS && "Could not get operand 1");
  auto *LHSVecTy = dyn_cast<FixedVectorType>(LHS->getType());
  VECZ_FAIL_IF(!LHSVecTy);
  const unsigned SrcWidth = LHSVecTy->getNumElements();
  const unsigned DstWidth = VecTy->getNumElements();

  // Determine which lanes we need from both vector operands.
  PacketMask LHSMask;
  PacketMask RHSMask;
  for (unsigned i = 0; i < DstWidth; i++) {
    if (!PM.isEnabled(i)) {
      continue;
    }
    int MaskLane = Shuffle->getMaskValue(i);
    if (MaskLane >= static_cast<int>(SrcWidth)) {
      MaskLane -= static_cast<int>(SrcWidth);
      RHSMask.enable(static_cast<unsigned>(MaskLane));
    } else if (MaskLane >= 0) {
      LHSMask.enable(static_cast<unsigned>(MaskLane));
    }
  }

  // Scalarize each vector operand as needed.
  const SimdPacket *LHSPacket = nullptr;
  if (LHSMask.Value != 0) {
    LHSPacket = scalarize(LHS, LHSMask);
    VECZ_FAIL_IF(!LHSPacket);
  }
  const SimdPacket *RHSPacket = nullptr;
  if (RHSMask.Value != 0) {
    RHSPacket = scalarize(RHS, RHSMask);
    VECZ_FAIL_IF(!RHSPacket);
  }

  // Copy the scalarized values to the result packet.
  SimdPacket *P = getPacket(Shuffle, DstWidth);
  for (unsigned i = 0; i < DstWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }
    Value *Extracted = nullptr;
    int MaskLane = Shuffle->getMaskValue(i);
    if (MaskLane < 0) {
      Extracted = UndefValue::get(VecTy->getElementType());
    } else if (MaskLane >= (int)SrcWidth) {
      MaskLane -= (int)SrcWidth;
      if (RHSPacket) {
        Extracted = RHSPacket->at(MaskLane);
      }
    } else if (MaskLane >= 0) {
      if (LHSPacket) {
        Extracted = LHSPacket->at(MaskLane);
      }
    }
    P->set(i, Extracted);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizeInsertElement(InsertElementInst *Insert,
                                               PacketMask PM) {
  Value *Vec = Insert->getOperand(0);
  VECZ_FAIL_IF(!Vec);
  Value *Ele = Insert->getOperand(1);
  assert(Ele && "Could not get operand 1 of Insert");
  Value *Index = Insert->getOperand(2);
  assert(Index && "Could not get operand 2 of Insert");
  const ConstantInt *CIndex = dyn_cast<ConstantInt>(Index);
  const auto *VecTy = cast<FixedVectorType>(Vec->getType());
  const unsigned IndexInt = CIndex ? CIndex->getZExtValue() : 0;
  const unsigned SimdWidth = VecTy->getNumElements();

  SimdPacket *P = getPacket(Insert, SimdWidth);

  // Scalarize the vector operand
  PacketMask OpMask;
  OpMask.enableAll(SimdWidth);
  // If we have a constant mask, we can skip the lane we are not going to use
  if (CIndex) {
    OpMask.disable(IndexInt);
  }
  const SimdPacket *VecP = scalarize(Vec, OpMask);
  VECZ_FAIL_IF(!VecP);

  // For each lane, we need to select either the original vector element (from
  // VecP) or the new value Ele. The selection is done based on the Index.
  IRBuilder<> B(Insert);
  for (unsigned lane = 0; lane < SimdWidth; ++lane) {
    if (!PM.isEnabled(lane) || P->at(lane)) {
      continue;
    }
    Value *LaneValue = nullptr;
    if (CIndex) {
      // If the Index is a Constant, then we can do the selection at compile
      // time
      LaneValue = (IndexInt == lane) ? Ele : VecP->at(lane);
    } else {
      // If the Index is a runtime value, then we have to emit select
      // instructions to do selection at runtime
      Constant *LaneC = ConstantInt::get(Index->getType(), lane);
      LaneValue =
          B.CreateSelect(B.CreateICmpEQ(Index, LaneC), Ele, VecP->at(lane));
    }
    P->set(lane, LaneValue);
  }

  return P;
}

SimdPacket *Scalarizer::scalarizeGEP(GetElementPtrInst *GEP, PacketMask PM) {
  auto *const vecDataTy = dyn_cast<FixedVectorType>(GEP->getType());
  VECZ_FAIL_IF(!vecDataTy);
  const unsigned simdWidth = vecDataTy->getNumElements();

  Value *const ptr = GEP->getPointerOperand();
  const SimdPacket *ptrPacket = nullptr;
  if (ptr->getType()->isVectorTy()) {
    ptrPacket = scalarize(ptr, PM);
    VECZ_FAIL_IF(!ptrPacket);
  }

  // Scalarize any vector GEP indices.
  SmallVector<SimdPacket *, 4> indexPackets;
  for (unsigned i = 0, n = GEP->getNumIndices(); i < n; ++i) {
    Value *const idx = GEP->getOperand(1 + i);
    if (idx->getType()->isVectorTy()) {
      SimdPacket *idxP = scalarize(idx, PM);
      VECZ_FAIL_IF(!idxP);
      indexPackets.push_back(idxP);
    } else {
      indexPackets.push_back(nullptr);
    }
  }

  IRBuilder<> B(GEP);
  const bool inBounds = GEP->isInBounds();
  const auto name = GEP->getName();
  SimdPacket *const P = getPacket(GEP, simdWidth);
  for (unsigned i = 0; i < simdWidth; i++) {
    if (!PM.isEnabled(i) || P->at(i)) {
      continue;
    }

    // Get the GEP indices per lane, scalarized or otherwise
    SmallVector<Value *, 4> scalarIndices;
    unsigned indexN = 1U;
    for (auto *idx : indexPackets) {
      if (idx) {
        scalarIndices.push_back(idx->at(i));
      } else {
        scalarIndices.push_back(GEP->getOperand(indexN));
      }
      ++indexN;
    }

    auto *const scalarPointer = ptrPacket ? ptrPacket->at(i) : ptr;
    Value *const newGEP =
        inBounds ? B.CreateInBoundsGEP(GEP->getSourceElementType(),
                                       scalarPointer, scalarIndices, name)
                 : B.CreateGEP(GEP->getSourceElementType(), scalarPointer,
                               scalarIndices, name);

    P->set(i, newGEP);
  }
  return P;
}

SimdPacket *Scalarizer::scalarizePHI(PHINode *Phi, PacketMask PM) {
  auto *PhiTy = cast<FixedVectorType>(Phi->getType());
  const unsigned Width = PhiTy->getNumElements();
  const unsigned NumIncoming = Phi->getNumIncomingValues();
  SmallVector<SimdPacket *, 2> Incoming;

  SimdPacket *P = getPacket(Phi, Width);
  IRBuilder<> B(Phi);

  SmallVector<unsigned, 4> ActiveLanes;

  // Start by creating the Phi nodes. This is done before everything else
  // because the IR might contain cycles which will cause the scalarization to
  // loop back to this Phi node when scalarizing the incoming values.
  for (unsigned lane = 0; lane < Width; ++lane) {
    if (!PM.isEnabled(lane) || P->at(lane)) {
      continue;
    }
    PHINode *SPhi =
        B.CreatePHI(PhiTy->getElementType(), NumIncoming, Phi->getName());
    P->set(lane, SPhi);
    ActiveLanes.push_back(lane);
  }

  // Scalarize the incoming values
  for (auto &In : Phi->incoming_values()) {
    SimdPacket *SIn = scalarize(In, PM);
    VECZ_FAIL_IF(!SIn);
    Incoming.push_back(SIn);
  }

  // Assign the scalarized incoming values to the scalarized Phi nodes
  for (const unsigned lane : ActiveLanes) {
    VECZ_ERROR_IF(!PM.isEnabled(lane), "Active lane should be enabled.");
    PHINode *SPhi = cast<PHINode>(P->at(lane));
    for (unsigned i = 0; i < NumIncoming; ++i) {
      SPhi->addIncoming(Incoming[i]->at(lane), Phi->getIncomingBlock(i));
    }
  }

  return P;
}
