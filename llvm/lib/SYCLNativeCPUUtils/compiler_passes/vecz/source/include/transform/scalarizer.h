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

/// @file
///
/// @brief Function scalarizer.

#ifndef VECZ_TRANSFORM_SCALARIZER_H_INCLUDED
#define VECZ_TRANSFORM_SCALARIZER_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <multi_llvm/llvm_version.h>

#include <vector>

#include "debugging.h"
#include "ir_cleanup.h"
#include "simd_packet.h"

namespace llvm {
class Instruction;
class LoadInst;
class StoreInst;
class CastInst;
class BitCastInst;
class BinaryOperator;
class FreezeInst;
class GetElementPtrInst;
class UnaryOperator;
class ICmpInst;
class FCmpInst;
class SelectInst;
class CallInst;
class ShuffleVectorInst;
class InsertElementInst;
class PHINode;
class ExtractElementInst;
class IntrinsicInst;
}  // namespace llvm

namespace vecz {

class VectorizationChoices;
class VectorizationContext;
struct MemOp;
struct PacketMask;
struct SimdPacket;

/// \addtogroup scalarization Scalarization Stage
/// @{
/// \ingroup vecz

/// @brief Holds the result of scalarization analysis for a given function.
class Scalarizer {
 public:
  /// @brief Create new scalarization results for the function.
  ///
  /// @param[in] F Function to scalarize.
  /// @param[in] Ctx VectorizationContext for this Function.
  /// @param[in] DoubleSuport True if double-precision floating point is
  /// supported
  Scalarizer(llvm::Function &F, VectorizationContext &Ctx, bool DoubleSuport);

  /// @brief Mark the value as needing scalarization.
  /// @param[in] V Value that needs scalarization.
  void setNeedsScalarization(llvm::Value *V);

  /// @brief Scalarize everything that has been marked for scalarization
  bool scalarizeAll();

  /// @brief A container type for instructions that failed to scalarize
  using FailureSet = llvm::DenseSet<const llvm::Value *>;

  /// @brief Get the list of instructions that failed to scalarize
  const FailureSet &failures() const { return Failures; }

 private:
  /// @brief Vectorization context for the function to scalarize.
  VectorizationContext &Ctx;
  llvm::Function &F;
  IRCleanup IC;
  bool DoubleSupport;

  /// @brief The values to scalarize, in order
  std::vector<llvm::Value *> ToScalarize;

  /// @brief The un-ordered set of values to scalarize for fast lookup
  llvm::DenseSet<llvm::Value *> ScalarizeSet;

  /// @brief Map of values to a gather of their scalarized elements
  llvm::DenseMap<llvm::Value *, llvm::Value *> Gathers;

  /// @brief Map onto packetized versions of scalar values
  llvm::DenseMap<const llvm::Value *, std::unique_ptr<SimdPacket>> packets;

  /// @brief The number of instructions that failed to scalarize
  FailureSet Failures;

  /// @brief Transform values that have non-vector types and vector operands
  /// by scalarizing their operands.
  ///
  /// @param[in] I Instruction whose operands to scalarize.
  ///
  /// @return A different value than V if the operands were scalarized; null if
  /// scalarization failed; or V if the value has no vector operand.
  llvm::Value *scalarizeOperands(llvm::Instruction *I);

  /// @brief Scalarize the given value from the function. Multiple calls to this
  /// function with the same value should return a cached result.
  ///
  /// @param[in] V Value to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarize(llvm::Value *V, PacketMask PM);

  /// @brief Get or create a packet for the given value.
  ///
  /// @param[in] V Value to retrieve a packet for.
  /// @param[in] SimdWidth Number of lanes in the packet.
  /// @param[in] Create true if a packet should be created if not present.
  ///
  /// @return SIMD packet for the given value.
  SimdPacket *getPacket(const llvm::Value *V, unsigned SimdWidth,
                        bool Create = true);

  llvm::Value *getGather(llvm::Value *V);

  /// @brief Perform post-scalarization tasks for the given value.
  ///
  /// @param[in] P Packet resulting from scalarization or null.
  /// @param[in] V Value to scalarize.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *assignScalar(SimdPacket *P, llvm::Value *V);
  /// @brief Extract an element's values, for use by scalarized users
  ///
  /// @param[in] V Value to extract.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *extractLanes(llvm::Value *V, PacketMask PM);
  /// @brief Scalarize a load instruction.
  ///
  /// @param[in] Load Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeLoad(llvm::LoadInst *Load, PacketMask PM);
  /// @brief Scalarize a store instruction.
  ///
  /// @param[in] Store Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeStore(llvm::StoreInst *Store, PacketMask PM);
  /// @brief Scalarize a cast instruction.
  ///
  /// @param[in] CastI Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeCast(llvm::CastInst *CastI, PacketMask PM);
  /// @brief Scalarize a bitcast instruction.
  ///
  /// @param[in] BC Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeBitCast(llvm::BitCastInst *BC, PacketMask PM);
  /// @brief Scalarize a binary operation instruction.
  ///
  /// @param[in] BinOp Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeBinaryOp(llvm::BinaryOperator *BinOp, PacketMask PM);
// Freeze instruction is not available in LLVM versions prior 10.0
// and not used in LLVM versions prior to 11.0
  /// @brief Scalarize a freeze instruction.
  ///
  /// @param[in] FreezeInst Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeFreeze(llvm::FreezeInst *FreezeI, PacketMask PM);
  /// @brief Scalarize a unary operation instruction.
  ///
  /// @param[in] UnOp Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeUnaryOp(llvm::UnaryOperator *UnOp, PacketMask PM);
  /// @brief Scalarize an interger compare instruction.
  ///
  /// @param[in] ICmp Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeICmp(llvm::ICmpInst *ICmp, PacketMask PM);
  /// @brief Scalarize a floating-point compare instruction.
  ///
  /// @param[in] FCmp Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeFCmp(llvm::FCmpInst *FCmp, PacketMask PM);
  /// @brief Scalarize a select instruction.
  ///
  /// @param[in] Select Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeSelect(llvm::SelectInst *Select, PacketMask PM);
  /// @brief Scalarize a call instruction.
  ///
  /// @param[in] CI Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeCall(llvm::CallInst *CI, PacketMask PM);
  /// @brief Scalarize a call instruction to a masked mem op.
  ///
  /// @param[in] CI Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  /// @param[in] MaskedOp Masked memory operation to scalarize.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeMaskedMemOp(llvm::CallInst *CI, PacketMask PM,
                                   MemOp &MaskedOp);
  /// @brief Scalarize a shuffle vector instruction.
  ///
  /// @param[in] Shuffle Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeShuffleVector(llvm::ShuffleVectorInst *Shuffle,
                                     PacketMask PM);
  /// @brief Scalarize an insert element instruction.
  ///
  /// @param[in] Insert Instruction to scalarize.
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return Packet containing scalarized values or null.
  SimdPacket *scalarizeInsertElement(llvm::InsertElementInst *Insert,
                                     PacketMask PM);
  /// @brief Scalarize GEPs with vector arguments
  ///
  /// @param[in] GEP The GEP to scalarize
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return The packet containing the scalarized values or null
  SimdPacket *scalarizeGEP(llvm::GetElementPtrInst *GEP, PacketMask PM);
  /// @brief Scalarize Phi nodes with vector arguments
  ///
  /// @param[in] Phi The Phi node to scalarize
  /// @param[in] PM Mask indicating which lanes are required.
  ///
  /// @return The packet containing the scalarized values or null
  SimdPacket *scalarizePHI(llvm::PHINode *Phi, PacketMask PM);
  /// @brief Preserves debug information attached to old instruction
  ///        we have just scalarized before it is removed.
  ///
  /// @param[in] Original Vector instruction which has been scalarized.
  /// @param[in] Packet Packetized instruction after scalarization.
  /// @param[in] Width SIMD width of packet.
  void scalarizeDI(llvm::Instruction *Original, const SimdPacket *Packet,
                   unsigned Width);

  // These functions work on scalar values that use vector values.

  /// @brief Scalarize the operands of an extract element instruction.
  ///
  /// @param[in] Extr Instruction to scalarize.
  ///
  /// @return A different value than Extr if the operands were scalarized; null
  /// if scalarization failed; or Extr if the value has no vector operand.
  llvm::Value *scalarizeOperandsExtractElement(llvm::ExtractElementInst *Extr);
  /// @brief Scalarize the operands of a bitcast instruction.
  ///
  /// @param[in] BC Instruction to scalarize.
  ///
  /// @return A different value than BC if the operands were scalarized; null if
  /// scalarization failed; or BC if the value has no vector operand.
  llvm::Value *scalarizeOperandsBitCast(llvm::BitCastInst *BC);

  /// @brief Scalarize the operands of a printf call.
  ///
  /// @param[in] CI Instruction to scalarize.
  ///
  /// @return A different value than CI if the operands were scalarized;
  /// null if scalarization failed; or CI if the value has no vector
  /// operand.
  llvm::Value *scalarizeOperandsPrintf(llvm::CallInst *CI);

  /// @brief Scalarize the operands of a binary operation instruction.
  ///
  /// @param[in] Intrin Instruction to scalarize.
  ///
  /// @return A different value than Intrin if the operands were scalarized;
  /// null if scalarization failed; or Intrin if the value has no vector
  /// operand.
  llvm::Value *scalarizeReduceIntrinsic(llvm::IntrinsicInst *Intrin);
};

/// @}
}  // namespace vecz

#endif  // VECZ_TRANSFORM_SCALARIZER_H_INCLUDED
