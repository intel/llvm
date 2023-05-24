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
/// @brief Function instantiator.

#ifndef VECZ_TRANSFORM_INSTANTIATION_PASS_H_INCLUDED
#define VECZ_TRANSFORM_INSTANTIATION_PASS_H_INCLUDED

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

namespace vecz {

class Packetizer;
class VectorizationContext;
class PacketRange;
struct MemOp;

/// @brief Instantiation pass where instructions that need it (vector or not)
/// are instantiated (i.e. duplicated with lane ID substitution), starting from
/// the leaves.
class InstantiationPass {
 public:
  /// @brief Create a new instantiation pass.
  ///
  /// @param[in] PP The packetizer object to call back to when required.
  InstantiationPass(Packetizer &PP);

  /// @brief Instantiate the given value from the function.
  /// The returned value is equivalent to a clone of the V 'expression' with any
  /// work-item ID (e.g. from get_global_id) adjusted with the lane's ID.
  ///
  /// @param[in] V Value to instantiate.
  ///
  /// @return Instantiated value.
  PacketRange instantiate(llvm::Value *V);

 private:
  /// @brief Duplicates an instruction across all SIMD Lanes.
  ///
  /// @param[in] I The instruction to duplicate across lanes
  ///
  /// @return The SIMD Packet
  PacketRange instantiateByCloning(llvm::Instruction *I);
  /// @brief Broadcasts an instruction across all SIMD Lanes.
  ///
  /// @param[in] I The instruction to extract elements from
  ///
  /// @return The SIMD Packet
  PacketRange simdBroadcast(llvm::Instruction *I);
  /// @brief Instantiate the given value from the function.
  /// The returned value is equivalent to a clone of the V 'expression' with any
  /// work-item ID (e.g. from get_global_id) adjusted with the lane's ID.
  ///
  /// @param[in] V Value to instantiate.
  ///
  /// @return Instantiated value.
  PacketRange instantiateInternal(llvm::Value *V);
  /// @brief Instantiate the given intruction from the function.
  /// The returned value is equivalent to a clone of the V 'expression' with any
  /// work-item ID (e.g. from get_global_id) adjusted with the lane's ID.
  ///
  /// @param[in] Ins instruction to instantiate.
  ///
  /// @return Instantiated value.
  PacketRange instantiateInstruction(llvm::Instruction *Ins);
  /// @brief Perform post-instantiation tasks.
  ///
  /// @param[in] P Packet that is the result of instantiation or null.
  /// @param[in] V Value that was instantiated.
  ///
  /// @return Instantiated packet or null.
  PacketRange assignInstance(const PacketRange P, llvm::Value *V);
  /// @brief Create a packet where all lanes contain the same value.
  ///
  /// @param[in] V Value to broadcast.
  ///
  /// @return Packet with the broadcasted value.
  PacketRange broadcast(llvm::Value *V);
  /// @brief Instantiate a call instruction.
  ///
  /// @param[in] CI Instruction to instantiate.
  ///
  /// @return Instantiated packet for the given instruction.
  PacketRange instantiateCall(llvm::CallInst *CI);
  /// @brief Instantiate an alloca instruction.
  ///
  /// @param[in] Alloca Instruction to instantiate.
  ///
  /// @return Instantiated packet for the given instruction.
  PacketRange instantiateAlloca(llvm::AllocaInst *Alloca);

  VectorizationContext &Ctx;
  Packetizer &packetizer;
};
}  // namespace vecz

#endif  // VECZ_TRANSFORM_INSTANTIATION_PASS_H_INCLUDED
