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
/// @brief Function packetizer helper classes.

#ifndef VECZ_TRANSFORM_PACKETIZATION_HELPERS_H_INCLUDED
#define VECZ_TRANSFORM_PACKETIZATION_HELPERS_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <multi_llvm/llvm_version.h>
#include <multi_llvm/multi_llvm.h>

#include <memory>

namespace llvm {
class Value;
class ShuffleVectorInst;
class Twine;
}  // namespace llvm

namespace vecz {
class TargetInfo;
struct SimdPacket;

/// @brief Determines the insertion point after the value V. If V has a position
/// in the function, (e.g., an Instruction), this method will return an
/// IRBuilder set to the next point after that. If V has no position (e.g., a
/// Constant or an Argument) then this method will return an IRBuilder set to a
/// suitable insertion point at the beginning of the function.
///
/// @param[in] V Value to insert instructions after, if an llvm::Instruction.
/// @param[in] F Function to insert instructions into, if V is not an
/// llvm::Instruction.
/// @param[in] IsPhi true if the instructions to insert are phis, false if the
/// insertion point should be after all phis in the basic block.
///
/// @return IRBuilder set to a suitable insertion point.
llvm::IRBuilder<> buildAfter(llvm::Value *V, llvm::Function &F,
                             bool IsPhi = false);

/// @brief Utility function for building a shufflevector instruction, absorbing
/// its operands where possible.
///
/// @param[in] B IRBuilder to build any new instruction created
/// @param[in] srcA the first vector operand of the new shuffle
/// @param[in] srcB the second vector operand of the new shuffle
/// @param[in] mask the shuffle mask
/// @param[in] name the name of the new instruction
///
/// @return a value identical to the requested shufflevector
llvm::Value *createOptimalShuffle(llvm::IRBuilder<> &B, llvm::Value *srcA,
                                  llvm::Value *srcB,
                                  const llvm::SmallVectorImpl<int> &mask,
                                  const llvm::Twine &name = llvm::Twine());

/// @brief Utility function for splatting a vector of scalars to create a
/// "vector of vectors", being the concatenation of vector splats of its
/// elements. eg. subSplat("ABCD", 4) == "AAAABBBBCCCCDDDD"
///
/// Only works on fixed vector types.
///
/// @param[in] TI TargetInfo for target-dependent optimizations
/// @param[in] B IRBuilder to build any new instructions created
/// @param[in,out] srcs The packet of vectors to sub-splat
/// @param[in] subWidth The width of the individual splats
///
/// @return true on success
bool createSubSplats(const vecz::TargetInfo &TI, llvm::IRBuilder<> &B,
                     llvm::SmallVectorImpl<llvm::Value *> &srcs,
                     unsigned subWidth);

/// @brief Utility function for creating a reduction operation.
///
/// The value must be a vector.
///
/// If VL is passed and is non-null, it is assumed to be the i32 value
/// representing the active vector length. The reduction will be
/// vector-predicated according to this length.
///
/// Only works on RecurKind::And, Or, Xor, Add, Mul, FAdd, FMul, {S,U,F}Min,
/// {S,U,F}Max.
llvm::Value *createMaybeVPTargetReduction(llvm::IRBuilderBase &B,
                                          const llvm::TargetTransformInfo &TTI,
                                          llvm::Value *Val,
                                          llvm::RecurKind Kind,
                                          llvm::Value *VL = nullptr);

/// @brief Utility function to obtain an indices vector to be used in a gather
/// operation.
///
/// When accessing a vector using an indices vector, this must be
/// modified taking into account the SIMD width.
///
/// @return An indices vector to be used in a gather operation; nullptr for LLVM
/// version < 13.
///
/// @param[in] B IRBuilder to build any new instructions created
/// @param[in] Indices Original indices vector
/// @param[in] Ty Type of the output vector
/// @param[in] FixedVecElts Original vector length
/// @param[in] N Name of the output variable
llvm::Value *getGatherIndicesVector(llvm::IRBuilder<> &B, llvm::Value *Indices,
                                    llvm::Type *Ty, unsigned FixedVecElts,
                                    const llvm::Twine &N = "");

/// @brief Returns a boolean vector with all elements set to 'true'.
llvm::Value *createAllTrueMask(llvm::IRBuilderBase &B, llvm::ElementCount EC);

/// @brief Returns an integer step vector, representing the sequence 0 ... N-1.
llvm::Value *createIndexSequence(llvm::IRBuilder<> &Builder,
                                 llvm::VectorType *VecTy,
                                 const llvm::Twine &Name = "");

/// @brief Class that represents a range in a vector of Value pointers.
/// The range is represented by its integer starting index and length, so that
/// it remains valid if the vector re-allocates its storage.
class PacketRange {
 public:
  using value_type = llvm::Value *;
  using iterator = value_type *;
  using const_iterator = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;

  /// @brief Construct an empty range
  constexpr PacketRange(std::vector<llvm::Value *> &d)
      : data(d), start(0), length(0) {}
  /// @brief Construct a range with given start index and length
  constexpr PacketRange(std::vector<llvm::Value *> &d, size_t s, size_t l)
      : data(d), start(s), length(l) {}

  /// @brief Copy constructor
  constexpr PacketRange(const PacketRange &) = default;
  /// @brief Move constructor
  constexpr PacketRange(PacketRange &&) = default;
  /// @brief Destructor
  ~PacketRange() = default;

  /// @brief Return the length of the range
  size_t size() const { return length; }
  /// @brief Standard container begin iterator
  iterator begin() { return &*data.begin() + start; }
  /// @brief Standard container begin const iterator
  const_iterator begin() const { return &*data.begin() + start; }
  /// @brief Standard container end iterator
  iterator end() { return begin() + length; }
  /// @brief Standard container end const iterator
  const_iterator end() const { return begin() + length; }
  /// @brief Return a reference to the element at given index
  reference at(size_t i) { return data[start + i]; }
  /// @brief Return a const reference to the element at given index
  const_reference at(size_t i) const { return data[start + i]; }
  /// @brief Return a reference to the element at given index
  reference operator[](size_t i) { return at(i); }
  /// @brief Return a const reference to the element at given index
  const_reference operator[](size_t i) const { return at(i); }
  /// @brief Return a reference to the first element in the range
  reference front() { return data[start]; }
  /// @brief Return a const reference to the first element in the range
  const_reference front() const { return data[start]; }
  /// @brief Return a reference to the last element in the range
  reference back() { return data[start + length - 1]; }
  /// @brief Return a const reference to the last element in the range
  const_reference back() const { return data[start + length - 1]; }

  /// @brief Convert to bool
  /// @returns false if length is zero, true otherwise
  operator bool() const { return length != 0; }

 private:
  std::vector<llvm::Value *> &data;
  const size_t start;
  const size_t length;
};

/// @brief Structure to hold the strategy-agnostic result of packetizing an
/// instruction (i.e. can represent either a vectorized or an instantiated
/// value) that enables the result to be converted on demand.
struct PacketInfo {
  /// @brief The number of instances created during packetization
  unsigned numInstances = 0;

  /// @brief Vectorized value. Each element in the vector represents a scalar
  /// instance (SIMD lane).
  llvm::Value *vector = nullptr;

  /// @brief Map of vector widths to packet range start indices
  llvm::SmallDenseMap<unsigned, unsigned, 2> packets;

  /// @brief Default constructor
  PacketInfo() = default;
  /// @brief Deleted copy constructor
  PacketInfo(const PacketInfo &) = delete;
  /// @brief Move constructor
  PacketInfo(PacketInfo &&) = default;
  /// @brief Destructor
  ~PacketInfo() = default;
  /// @brief Deleted copy assignment operator
  PacketInfo &operator=(const PacketInfo &) = delete;
  /// @brief Move assignment operator
  PacketInfo &operator=(PacketInfo &&) = default;

  /// @brief get the range of values for a given packet width
  PacketRange getRange(std::vector<llvm::Value *> &d, unsigned width) const;

  /// @brief get the range of values for the originally created packet.
  PacketRange getRange(std::vector<llvm::Value *> &d) const {
    return getRange(d, numInstances);
  }
};

}  // namespace vecz

#endif  // VECZ_TRANSFORM_PACKETIZATION_HELPERS_H_INCLUDED
