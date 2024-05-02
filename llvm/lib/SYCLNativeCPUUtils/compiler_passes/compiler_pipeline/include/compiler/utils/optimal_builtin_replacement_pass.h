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
/// Optimal builtin replacement pass.

#ifndef COMPILER_UTILS_OPTIMAL_BUILTIN_REPLACEMENT_PASS_H_INCLUDED
#define COMPILER_UTILS_OPTIMAL_BUILTIN_REPLACEMENT_PASS_H_INCLUDED

#include <compiler/utils/mangling.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LazyCallGraph.h>
#include <llvm/IR/PassManager.h>

namespace compiler {
namespace utils {

/// @brief A Callgraph optimization pass which replaces calls to builtin
/// functions with more optimal versions, either via inlined code, or calls to
/// suitable llvm intrinsics which will later be lowered to optimal machine
/// code. When run with a non-null BuiltinInfo analysis, the builtin info is
/// queried to determine the properties of each call in the graph.
///
/// A set of replacement functions with identical signatures is kept by this
/// pass. These are invoked in order one after another on each call instruction
/// in the call graph. If any replacement returns a non-null `Value*` it is
/// used to replace the call and no further replacements are attempted on that
/// call. It is assumed that no replacement introduces new calls to the graph.
/// The set of replacements can be modified by users by setting
/// `adjustReplacements`.
///
/// The default set of replacement functions, in order, is:
/// * replaceAbacusCLZ
/// * replaceAbacusMulhi
/// * replaceAbacusFMinFMax
/// * Invoking emitBuiltinInline from BuiltinInfo analysis
class OptimalBuiltinReplacementPass
    : public llvm::PassInfoMixin<OptimalBuiltinReplacementPass> {
 public:
  using ReplacementFnTy = std::function<llvm::Value *(
      llvm::CallBase &, llvm::StringRef,
      const llvm::SmallVectorImpl<llvm::Type *> &,
      const llvm::SmallVectorImpl<compiler::utils::TypeQualifiers> &)>;

  /// @brief Constructor. Sets up default builtin replacements.
  OptimalBuiltinReplacementPass();

  llvm::PreservedAnalyses run(llvm::LazyCallGraph::SCC &C,
                              llvm::CGSCCAnalysisManager &AM,
                              llvm::LazyCallGraph &CG,
                              llvm::CGSCCUpdateResult &UR);

  /// @brief A callback invoked per-SCC before any replacements are performed,
  /// allowing customization of the replacements to be performed. The default
  /// set of replacements are passed in and may be modified in any way.
  std::function<void(std::vector<ReplacementFnTy> &)> adjustReplacements;

  /// @brief Replaces calls __abacus_clz(ty) with @llvm.ctlz(ty, i1 false)
  /// indicating that zero does not produce a poison result.
  /// Note: This replacement is not performend on 64-bit scalar or vectors of
  /// 64-bit scalar types.
  static llvm::Value *replaceAbacusCLZ(
      llvm::CallBase &CB, llvm::StringRef,
      const llvm::SmallVectorImpl<llvm::Type *> &,
      const llvm::SmallVectorImpl<compiler::utils::TypeQualifiers> &);

  /// @brief Replaces __abacus_mul_hi(ty lhs, ty rhs) with a sequence:
  ///   %lhs.ext = ext ty %lhs to x2bw(ty)
  ///   %rhs.ext = ext ty %rhs to x2bw(ty)
  ///   %mul.ext = mul x2bw(ty) %lhs.ext, %rhs.ext
  ///   %lo.part = ashr x2bw(ty) %mul.ext, bw(ty)
  ///   %res = trunc x2bw(ty) %lo.part to ty
  /// Where x2bw(ty) returns a type with twice the (element) bit-width, and
  /// bw(ty) returns the bit-width of a (element) type as an integer.
  /// This pattern is better matched by LLVM and target backends often produce
  /// "mul_hi" instructions as a result.
  static llvm::Value *replaceAbacusMulhi(
      llvm::CallBase &, llvm::StringRef,
      const llvm::SmallVectorImpl<llvm::Type *> &,
      const llvm::SmallVectorImpl<compiler::utils::TypeQualifiers> &);

  /// @brief Replaces __abacus_(fmin|fmax)(ty1 lhs, ty2 rhs) with
  /// @llvm.(minnum|maxnum)(ty1 lhs, ty1 rhs), where ty2 may be a scalar type
  /// which is splatted to a vector of ty1, where appropriate.
  /// Note: This replacement is not performed on ARM or AArch64 targets, due to
  /// LLVM backend bugs (https://llvm.org/PR27363).
  static llvm::Value *replaceAbacusFMinFMax(
      llvm::CallBase &, llvm::StringRef,
      const llvm::SmallVectorImpl<llvm::Type *> &,
      const llvm::SmallVectorImpl<compiler::utils::TypeQualifiers> &);

 private:
  std::vector<ReplacementFnTy> replacements;

  llvm::Value *replaceBuiltinWithInlineIR(llvm::CallBase &CB) const;
};

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_OPTIMAL_BUILTIN_REPLACEMENT_PASS_H_INCLUDED
