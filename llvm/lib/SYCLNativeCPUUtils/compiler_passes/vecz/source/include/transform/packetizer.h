// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
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
/// @brief Function packetizer.

#ifndef VECZ_TRANSFORM_PACKETIZER_H_INCLUDED
#define VECZ_TRANSFORM_PACKETIZER_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/TypeSize.h>
#include <multi_llvm/llvm_version.h>

#include <memory>

#include "ir_cleanup.h"
#include "transform/packetization_helpers.h"

namespace vecz {

struct MemOp;
class InstantiationPass;
class PacketizationAnalysisResult;
class StrideAnalysisResult;
struct UniformValueResult;
class VectorizationUnit;
class VectorizationContext;
class VectorizationChoices;

/// \addtogroup packetization Packetization Stage
/// @{
/// \ingroup vecz

/// @brief The implementation of the packetization process
class Packetizer {
public:
  class Result {
    friend class Packetizer;

  public:
    Result() = delete;
    Result(const Result &) = default;
    constexpr Result(Result &&) = default;

    Result(Packetizer &p) : packetizer(p), scalar(nullptr), info(nullptr) {}
    Result(Packetizer &p, llvm::Value *s, PacketInfo *i)
        : packetizer(p), scalar(s), info(i) {}

    operator bool() const { return info; }

    /// @brief Get a packetized/instantiated instruction as a vector value.
    /// If the value was instantiated, this will construct and return a gather
    /// of the SIMD lanes.
    ///
    /// @return Packetized value
    llvm::Value *getAsValue() const;

    /// @brief Get a packetized/instantiated instruction as a SIMD packet.
    /// If the value was packetized, this will construct a new packet by
    /// extracting the elements.
    ///
    /// @param[in] width the width of the packet to get.
    ///
    /// @return Instantiated packet
    PacketRange getAsPacket(unsigned width) const;

    /// @brief Get a copy of all the Values from the vector or packet, as
    /// the width it was originally packetized to.
    ///
    /// @param[out] vals a vector of Values representing the result.
    void getPacketValues(llvm::SmallVectorImpl<llvm::Value *> &vals) const;

    /// @brief Get a copy of all the Values from the vector or packet.
    /// When `width == 1` this will return a length-1 result containing the
    /// vector valued result. Otherwise, it copies the values from the
    /// packet of the requested width.
    ///
    /// @param[in] width the width of the packet to get.
    /// @param[out] vals a vector of Values representing the result.
    void getPacketValues(unsigned width,
                         llvm::SmallVectorImpl<llvm::Value *> &vals) const;

  private:
    Packetizer &packetizer;
    llvm::Value *const scalar;
    PacketInfo *const info;

    PacketRange createPacket(unsigned width) const;
    PacketRange getRange(unsigned width) const;
    PacketRange widen(unsigned width) const;
    PacketRange narrow(unsigned width) const;
    const Result &broadcast(unsigned width) const;
  };

  /// @brief Packetize the given function, duplicating its behaviour (defined
  /// values and side effects) for each lane of a SIMD packet.
  ///
  /// @param[in] F Function to packetize.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  /// @param[in] Width the vectorization factor
  /// @param[in] Dim the vectorization dimension
  ///
  /// @return true if the function was packetized, false otherwise.
  static bool packetize(llvm::Function &F, llvm::FunctionAnalysisManager &AM,
                        llvm::ElementCount Width, unsigned Dim);

  /// @brief Packetize the given value from the function.
  ///
  /// @param[in] V Value to packetize.
  ///
  /// @return Packetized value.
  Result packetize(llvm::Value *V);

  /// @brief Return an already packetized value.
  ///
  /// @param[in] V Value to query.
  ///
  /// @return Packetized value or nullptr.
  Result getPacketized(llvm::Value *V);

  /// @brief Create a new SIMD packet to hold an instantiated value.
  ///
  /// @param[in] V the value the packet will represent
  /// @param[in] width the SIMD width of the packet
  ///
  /// @returns a new packet
  PacketRange createPacket(llvm::Value *V, unsigned width);

  /// @brief Get the Uniform Value Result
  ///
  /// @return the Uniform Value Result
  const UniformValueResult &uniform() const { return UVR; }

  /// @brief get the vectorization factor.
  llvm::ElementCount width() const { return SimdWidth; }

  /// @brief get the vectorization factor.
  unsigned dimension() const { return Dimension; }

  /// @brief get the function being packetized
  llvm::Function &function() { return F; }

  /// @brief get the Vectorization Context
  VectorizationContext &context() { return Ctx; }

  /// @brief get the Vectorization Context
  const VectorizationChoices &choices() const { return Choices; }

  PacketRange getEmptyRange() { return PacketRange(packetData); }

  /// @brief mark the instruction for deletion when packetization finishes
  void deleteInstructionLater(llvm::Instruction *I) {
    IC.deleteInstructionLater(I);
  }

private:
  Packetizer(llvm::Function &, llvm::FunctionAnalysisManager &AM,
             llvm::ElementCount Width, unsigned Dim);
  Packetizer() = delete;
  Packetizer(const Packetizer &) = delete;
  Packetizer(Packetizer &&) = delete;
  ~Packetizer() = default;

  llvm::FunctionAnalysisManager &AM;
  VectorizationUnit &VU;
  VectorizationContext &Ctx;
  const VectorizationChoices &Choices;
  UniformValueResult &UVR;
  StrideAnalysisResult &SAR;
  PacketizationAnalysisResult &PAR;
  llvm::Function &F;
  IRCleanup IC;

  /// @brief Vectorization factor
  llvm::ElementCount SimdWidth;

  /// @brief Vectorization dimension
  unsigned Dimension;

  /// @brief Map onto packetized versions of scalar values
  llvm::DenseMap<llvm::Value *, PacketInfo> packets;

  /// @brief Central storage for all the packetized values
  ///
  /// This vector is a contiguous storage for all the wide packets created
  /// during the packetization process. New packets get allocated to a
  /// range at the end of the vector, and are referenced by index so that
  /// they are not invalidated when the storage is re-allocated. Vector
  /// elements will never be erased during packetization, and the data will
  /// not be cleared until the packetizer itself is destroyed.
  /*
                 /^ ^\
     "No take"  / 0 0 \
                V\ Y /V  */
  std::vector<llvm::Value *> packetData;
  /*             |    \
                 || (__V  "ONLY GROW"
  */

  /// @brief The value representing the current (dynamic) active vector length
  /// for this kernel. This value is the *base* vector length for one scalar
  /// work-item; vector operations must be scaled according to their vector
  /// width.
  /// If non-null, packetized operations are required to respect this active
  /// length if they would produce side effects.
  llvm::Value *VL = nullptr;

  /// @brief This class contains the private implementation of the packetizer.
  /// Declaring it as an inner class of the Packetizer class allows it access
  /// to its private members (including its constructor).
  class Impl;
};

/// @}
} // namespace vecz

#endif // VECZ_TRANSFORM_PACKETIZER_H_INCLUDED
