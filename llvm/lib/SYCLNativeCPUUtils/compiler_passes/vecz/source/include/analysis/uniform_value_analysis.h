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
/// @brief Uniform Value analysis.

#ifndef VECZ_ANALYSIS_UNIFORM_VALUE_RANGE_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_UNIFORM_VALUE_RANGE_ANALYSIS_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>

#include <vector>

namespace llvm {
class Value;
class Instruction;
}  // namespace llvm

namespace vecz {

class VectorizationContext;
class VectorizationUnit;

/// @brief Holds the result of Uniform Value Analysis for a given function.
struct UniformValueResult {
  enum class VaryingKind {
    /// @brief The value is truly uniform on all active and inactive lanes.
    eValueTrueUniform,
    /// @brief The value is uniform on active lanes. May be poison or undefined
    /// on inactive lanes.
    eValueActiveUniform,
    /// @brief The value is varying and lanes may see different values.
    eValueVarying,
    /// @brief The value is uniform, but its mask is not.
    /// Used for masked memory operations with a uniform address but varying
    /// mask.
    eMaskVarying,
  };

  /// @brief The function the analysis was run on.
  llvm::Function &F;
  /// @brief Vectorization unit the analysis was run on.
  VectorizationUnit &VU;
  /// @brief The Vectorization Context of the analysis.
  VectorizationContext &Ctx;
  /// @brief The vectorization dimension
  unsigned dimension;
  /// @brief The actual results of the analysis.
  llvm::DenseMap<const llvm::Value *, VaryingKind> varying;

  /// @brief Create a new UVA result for the given unit.
  /// @param[in] F Function to analyze.
  /// @param[in] VU Function to analyze.
  UniformValueResult(llvm::Function &F, VectorizationUnit &VU);

  /// @brief Determine whether the given value needs to be packetized or not.
  ///
  /// @param[in] V Value to analyze.
  ///
  /// @return true if the value needs to be packetized, false otherwise.
  bool isVarying(const llvm::Value *V) const;

  /// @brief Determine whether the given value has a varying mask or not.
  ///
  /// @param[in] V Value to analyze.
  ///
  /// @return true if the value has a varying mask, false otherwise.
  bool isMaskVarying(const llvm::Value *V) const;

  /// @brief Determine whether the given value has a varying mask or not.
  ///
  /// @param[in] V Value to analyze.
  ///
  /// @return true if the value is varying or has a varying mask, false
  /// otherwise.
  bool isValueOrMaskVarying(const llvm::Value *V) const;

  /// @brief Determine (on demand) whether the given value is a true uniform
  /// value.
  ///
  /// @param[in] V Value to analyze.
  ///
  /// @return true if the value is true uniform, false otherwise. Caches the
  /// result for future queries.
  bool isTrueUniform(const llvm::Value *V);

  /// @brief Remove the value from the analysis.
  ///
  /// @param[in] V Value to remove.
  void remove(const llvm::Value *V) { varying.erase(V); }

  /// @brief Uncritically set a value to varying.
  /// This can be used to keep the result valid after expression transforms.
  /// Use with care, since it does not recursively update value users.
  ///
  /// @param[in] V Value to set.
  void setVarying(const llvm::Value *V) {
    varying[V] = VaryingKind::eValueVarying;
  }

  /// @brief Look for vector roots in the function.
  ///
  /// Roots are values which are scalar in the original function but are defined
  /// to be vector in the vectorized function.
  ///
  /// Users of roots need to be vectorized too but are not considered roots.
  /// As such they will not be returned in Roots.
  ///
  /// Examples:
  /// * Calls to get_global_id()
  /// * Calls to get_local_id()
  ///
  /// @param[in,out] Roots List of roots to update.
  void findVectorRoots(std::vector<llvm::Value *> &Roots) const;

  /// @brief Look for vector leaves in the function.
  ///
  /// Leaves are instructions that allow vectorized values to 'escape' from the
  /// function.
  ///
  /// Examples:
  /// * Store instructions (when the value to store is vectorized)
  /// * Operands of call instructions (when the call needs to be vectorized)
  /// * Return instructions
  ///
  /// @param[in,out] Leaves List of leaves to update.
  void findVectorLeaves(std::vector<llvm::Instruction *> &Leaves) const;

  /// @brief Find the alloca that this pointer points to
  ///
  /// @param[in] Pointer The pointer that is (potentially) pointing in an alloca
  ///
  /// @return the alloca if found, or nullptr otherwise
  static llvm::AllocaInst *findAllocaFromPointer(llvm::Value *Pointer);

  /// @brief Try to extract the base pointer of the address.
  ///
  /// @param[in] Address Address to split into base and offset.
  ///
  /// @return Base address.
  llvm::Value *extractMemBase(llvm::Value *Address);

  // private:
  /// @brief Mark any value in the function that depends on V as being varying.
  ///
  /// @param[in] V Value used to start the vectorization search.
  /// @param[in] From Optional value being used by `V`.
  void markVaryingValues(llvm::Value *V, llvm::Value *From = nullptr);
};

/// @brief Analysis that determine whether values in a function are uniform or
/// varying.
class UniformValueAnalysis
    : public llvm::AnalysisInfoMixin<UniformValueAnalysis> {
  friend AnalysisInfoMixin<UniformValueAnalysis>;

 public:
  /// @brief Create a new analysis object.
  UniformValueAnalysis() {}

  /// @brief Type of result produced by the analysis.
  using Result = UniformValueResult;

  /// @brief Determine which values in the function are uniform and which are
  /// potentially varying.
  ///
  /// @param[in] F Function to analyze.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Analysis result for the function.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Uniform value analysis"; }

 private:
  /// @brief Unique identifier for the pass.
  static llvm::AnalysisKey Key;
};

}  // namespace vecz

#endif  // VECZ_ANALYSIS_UNIFORM_VALUE_RANGE_ANALYSIS_H_INCLUDED
