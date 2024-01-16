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
/// @brief Analysis of memory pointer offsets.

#ifndef VECZ_OFFSET_INFO_H_INCLUDED
#define VECZ_OFFSET_INFO_H_INCLUDED

#include <inttypes.h>
#include <llvm/IR/IRBuilder.h>

namespace llvm {
class CallInst;
class Value;
class Type;
}  // namespace llvm

namespace vecz {

struct UniformValueResult;
class ValueTagMap;

/// @brief Item ID dependence kinds that an expression can have.
/// Note that these are all mutually exclusive.
enum OffsetKind {
  /// @brief The offset may diverge in unmodelled ways when vectorized. This
  /// state is to be assumed unless it can be proved otherwise.
  eOffsetMayDiverge,
  /// @brief The offset is a compile-time constant.
  eOffsetConstant,
  /// @brief The offset is a uniform variable.
  eOffsetUniformVariable,
  /// @brief The offset has a work-item ID dependence. The ID might be scaled
  /// by some stride != 1, in which case loads or stores dependent on it will
  /// be interleaved.
  eOffsetLinear
};

class StrideAnalysisResult;

/// @brief Describes an offset used by a load or store instruction we want to
/// vectorize.
struct OffsetInfo {
  /// @brief Properties of the offset, which may prevent vectorization.
  OffsetKind Kind;
  /// @brief The actual value of the analyzed expression.
  llvm::Value *const ActualValue;
  /// @brief The difference in this value between two consecutive work items,
  /// as a constant integer.
  /// When the stride is a pointer, the difference is in bytes.
  int64_t StrideInt;
  /// @brief The difference in this value between two consecutive work items,
  /// as a uniform value.
  /// When the stride is a pointer, the difference is in bytes.
  /// This is nullptr after analysis and is set upon calling `manifest()`.
  llvm::Value *ManifestStride;

  /// @brief A bit mask indicating which bits of the value it is possible to be
  /// set, based on the expressions it depends on.
  uint64_t BitMask;

  /// @brief Construct a new offset information object from a general value
  /// @param[in] B The StrideAnalysisResult used to retrieve other OffsetInfos.
  /// @param[in] V Offset value to analyze.
  OffsetInfo(StrideAnalysisResult &SAR, llvm::Value *V);

  OffsetInfo() = delete;
  OffsetInfo(const OffsetInfo &) = default;

  /// @brief Return whether the offset has a non-analytical dedpendence on work
  /// item ID.
  bool mayDiverge() const { return Kind == eOffsetMayDiverge; }

  /// @brief Return whether the offset has a linear dependence on work item ID.
  bool hasStride() const { return Kind == eOffsetLinear; }

  /// @brief Return whether the offset is a compile-time constant.
  bool isConstant() const { return Kind == eOffsetConstant; }

  /// @brief Return whether the offset has no dependence on work item ID.
  bool isUniform() const {
    return Kind == eOffsetConstant || Kind == eOffsetUniformVariable;
  }

  /// @brief Returns the actual value of the analyzed offset if it is uniform.
  ///
  /// @return The uniform Value or nullptr otherwise
  llvm::Value *getUniformValue() const;
  /// @brief Get the offset as a constant int. It assumes that it is possible to
  /// do so.
  /// @return The offset as an integer
  int64_t getValueAsConstantInt() const;
  /// @brief Get the Stride of the analyzed and manifested value.
  /// @return The stride in number of elements
  llvm::Value *getStride() const { return ManifestStride; }
  /// @brief Determine whether the stride is simply a constant compile time
  /// integer.
  /// @return true if the stride is linear and constant, false otherwise.
  bool isStrideConstantInt() const;
  /// @brief Get the stride as a constant int.
  /// @return The stride as an integer, or zero if the stride is not constant.
  int64_t getStrideAsConstantInt() const;

  /// @brief Convert the bytewise stride into an element-wise stride based on
  /// the data type and data layout, as an integer.
  ///
  /// @param[in] PtrTy The element data type.
  /// @param[in] DL The Data Layout.
  /// @return The memory stride as number of elements.

  uint64_t getConstantMemoryStride(llvm::Type *PtrEleTy,
                                   const llvm::DataLayout *DL) const;

  /// @brief Convert the bytewise stride into an element-wise stride based on
  /// the data type and data layout, building instructions where needed. Note
  /// that the stride must be manifest first.
  ///
  /// @param[in] B an IRBuilder used for creating constants or instructions.
  /// @param[in] PtrTy The element data type.
  /// @param[in] DL The Data Layout.
  /// @return The memory stride as number of elements.
  llvm::Value *buildMemoryStride(llvm::IRBuilder<> &B, llvm::Type *PtrEleTy,
                                 const llvm::DataLayout *DL) const;

  /// @brief Create Values that represent or compute strides.
  ///
  /// @param[in] B an IRBuilder used for creating constants or instructions.
  /// @return Reference to the current object for chaining.
  OffsetInfo &manifest(llvm::IRBuilder<> &B, StrideAnalysisResult &SAR);

 private:
  /// @brief Mark this offset with the given flag.
  /// @return Reference to the current object for chaining.
  OffsetInfo &setKind(OffsetKind Kind);
  /// @brief Mark this offset as having a stride component.
  /// @param[in] Stride Stride component applied to the item ID.
  /// @return Reference to the current object for chaining.
  OffsetInfo &setStride(llvm::Value *Stride);
  /// @brief Mark this offset as having a stride component.
  /// @param[in] Stride Stride component applied to the item ID.
  /// @return Reference to the current object for chaining.
  OffsetInfo &setStride(int64_t Stride);
  /// @brief Mark this offset as possibly diverging.
  /// @return Reference to the current object for chaining.
  OffsetInfo &setMayDiverge();

  /// @brief Analyse the given integer offset for properties that we need to
  /// know in order to vectorize loads and stores. In particular we are
  /// interested in knowing whether the offset can diverge (be different for
  /// different items) or not. We can handle divergence in several cases but not
  /// all.
  ///
  /// @param[in] Offset Offset value to analyze.
  /// @param[in] SAR Result of the stride analysis.
  ///
  /// @return Reference to the current object for chaining.
  OffsetInfo &analyze(llvm::Value *Offset, StrideAnalysisResult &SAR);

  /// @brief Analyse the given pointer for properties that we need to
  /// know in order to vectorize loads and stores. In particular we are
  /// interested in knowing whether the offset can diverge (be different for
  /// different items) or not. We can handle divergence in several cases but not
  /// all.
  ///
  /// @param[in] Address Pointer to analyze.
  /// @param[in] SAR Result of the stride analysis.
  ///
  /// @return Reference to the current object for chaining.
  OffsetInfo &analyzePtr(llvm::Value *Address, StrideAnalysisResult &SAR);

  /// @brief Combine the offset info of LHS and RHS operands of an add
  /// operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineAdd(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestAdd(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                          const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of a sub operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineSub(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestSub(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                          const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of an and
  /// operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineAnd(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestAnd(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                          const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of an or operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineOr(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestOr(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                         const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of a xor operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineXor(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestXor(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                          const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of a shl operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineShl(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestShl(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                          const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of a ashr
  /// operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineAShr(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestAShr(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                           const OffsetInfo &RHS);

  /// @brief Combine the offset info of LHS and RHS operands of a mul operation.
  /// @param[in] LHS Offset info for the LHS operand.
  /// @param[in] RHS Offset info for the RHS operand.
  /// @return Reference to the current object for chaining.
  OffsetInfo &combineMul(const OffsetInfo &LHS, const OffsetInfo &RHS);
  OffsetInfo &manifestMul(llvm::IRBuilder<> &B, const OffsetInfo &LHS,
                          const OffsetInfo &RHS);

  /// @brief Copies the stride information from another OffsetInfo into this one
  /// @param[in] Other the other OffsetInfo to copy from
  /// @return Reference to the current object for chaining.
  OffsetInfo &copyStrideFrom(const OffsetInfo &Other);
};

}  // namespace vecz

#endif  // #define VECZ_OFFSET_INFO_H_INCLUDED
