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

#ifndef VECZ_VECTORIZATION_UNIT_H_INCLUDED
#define VECZ_VECTORIZATION_UNIT_H_INCLUDED

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class Function;
class FunctionType;
class Module;
class Instruction;
class Argument;
} // namespace llvm

namespace vecz {
namespace internal {
struct VeczFailResult;
struct AnalysisFailResult;
} // namespace internal

struct SimdPacket;
struct UniformValueResult;
class ValueTagMap;
class VectorizationContext;
class VectorizationChoices;

template <typename T> class AnalysisWrapper;

/// @brief Describe an argument of a function that needs to be vectorized.
struct VectorizerTargetArgument {
  /// @brief Argument of the scalar function.
  llvm::Argument *OldArg;
  /// @brief Argument of the vectorized function. Might be scalar or vector.
  llvm::Argument *NewArg;
  /// @brief Whether the argument needs to be vectorized or not.
  bool IsVectorized;
  /// @brief If the argument is a 'byref' pointer used to return a value, this
  /// is the type of that value. Else it is null.
  llvm::Type *PointerRetPointeeTy;
  /// @brief Placeholder instruction for arguments needing vectorization.
  llvm::Instruction *Placeholder;
};

/// @brief Analysis flags that can be attached to LLVM functions.
enum FunctionFlags {
  eFunctionNoFlag = 0,
  /// @brief The function has been analyzed.
  /// Set by the preliminary vectorization analysis (canVectorize). Set once.
  eFunctionAnalysisDone = (1 << 0),
  /// @brief The function can be vectorized.
  /// Set by the preliminary vectorization analysis (canVectorize). Set once.
  eFunctionVectorizable = (1 << 1),
  /// @brief Vectorization of the function failed.
  /// Can be set by any pass. Set once.
  eFunctionVectorizationFailed = (1 << 2),
};

/// @brief struct to hold only the data needed to use a vectorized function
struct VectorizationResult {
  struct Arg {
    enum Kind { SCALAR, VECTORIZED, POINTER_RETURN } kind;
    llvm::Type *type;
    llvm::Type *pointerRetPointeeTy = nullptr;
    constexpr Arg(Kind k, llvm::Type *ty, llvm::Type *ptrRetTy)
        : kind(k), type(ty), pointerRetPointeeTy(ptrRetTy) {}
  };

  llvm::Function *func = nullptr;
  llvm::SmallVector<Arg, 2> args;

  operator bool() const { return func; }
  llvm::Function *get() const { return func; }
};

/// @brief Describe a function that needs to be vectorized.
class VectorizationUnit {
public:
  /// @brief Create a new vectorization unit for the given scalar function.
  ///
  /// @param[in] F Function to vectorize.
  /// @param[in] Width SIMD width (i.e. vectorization factor) to use.
  /// @param[in] Dimension SIMD dimension to use (0 => x, 1 => y, 2 => z).
  /// @param[in] Ctx Context for vectorization.
  /// @param[in] Ch Vectorization Choices for the vectorization.
  VectorizationUnit(llvm::Function &F, llvm::ElementCount Width,
                    unsigned Dimension, VectorizationContext &Ctx,
                    const VectorizationChoices &Ch);
  /// @brief Free up any resource used by the function.
  ~VectorizationUnit();

  /// @brief Access the vectorization context linked to this function.
  VectorizationContext &context() { return Ctx; }

  /// @brief Access the vectorization context linked to this function.
  const VectorizationContext &context() const { return Ctx; }

  /// @brief Number of available SIMD lanes, i.e. vectorization factor.
  llvm::ElementCount width() const { return SimdWidth; }

  /// @brief Get the work group size along the vectorization dimension.
  uint64_t getLocalSize() const { return LocalSize; }

  /// @brief Whether to run the SIMD Width Analysis during vectorization.
  bool autoWidth() const { return AutoSimdWidth; }

  /// @brief Index of SIMD dimension used in vectorization.
  unsigned dimension() const { return SimdDimIdx; }

  /// @brief Set the SIMD width, i.e. vectorization factor. After changing this
  /// value a possible existing vectorized function is looked up in the module.
  ///
  /// @param[in] NewWidth New SIMD width.
  void setWidth(llvm::ElementCount NewWidth);

  /// @brief Set the work group size along the vectorization dimension.
  ///
  /// @param[in] LS the local work group size
  void setLocalSize(uint64_t LS) { LocalSize = LS; }

  /// @brief Set whether to use the SIMD width analysis
  ///
  /// @param[in] Auto true to use auto SIMD width, false otherwise
  void setAutoWidth(bool Auto) { AutoSimdWidth = Auto; }

  /// @brief Determine whether vectorizing the function failed or not.
  bool failed() const { return hasFlag(eFunctionVectorizationFailed); }

  /// @brief Mark this function as failing vectorization.
  /// @param[in] Remark Message to print into the optimization remarks
  /// @param[in] F Function to pass to emitVeczRemarkMissed
  /// @param[in] V Value to pass to emitVeczRemarkMissed
  /// @return unconditionally returns a VeczFailResult which can be safely
  /// ignored. This can help cut down on some boilerplate in contexts where
  /// we'll immediately return, via the following idiom:
  /// ```
  ///   if (!thing) {
  ///     return setFailed("thing wasn't");
  ///   }
  /// ```
  internal::AnalysisFailResult setFailed(const char *Remark,
                                         const llvm::Function *F = nullptr,
                                         const llvm::Value *V = nullptr);

  /// @brief Check whether the function has the given flag or not.
  ///
  /// @param[in] Flag Flag to check.
  ///
  /// @return true if the function has the given flag, false otherwise.
  bool hasFlag(FunctionFlags Flag) const { return (FnFlags & Flag) == Flag; }

  /// @brief Set the given flag to the function.
  ///
  /// @param[in] Flag Flag to set.
  void setFlag(FunctionFlags Flag) {
    FnFlags = (FunctionFlags)(FnFlags | Flag);
  }

  /// @brief Clear the given flag from the function.
  ///
  /// @param[in] Flag Flag to set.
  void clearFlag(FunctionFlags Flag) {
    FnFlags = (FunctionFlags)(FnFlags & ~Flag);
  }

  /// @brief Access the arguments of the function to vectorize.
  const llvm::SmallVectorImpl<VectorizerTargetArgument> &arguments() const {
    return Arguments;
  }

  /// @brief Return the vectorized function if it exists, otherwise the original
  /// function.
  llvm::Function &function();

  /// @brief Return the vectorized function if it exists, otherwise the original
  /// function.
  const llvm::Function &function() const;

  /// @brief Original function to vectorize.
  llvm::Function *scalarFunction() const { return ScalarFn; }

  /// @brief Set the function to vectorize. This updates the function arguments.
  ///
  /// @param[in] NewFunction Original function.
  void setScalarFunction(llvm::Function *NewFunction);

  /// @brief Vectorized function.
  llvm::Function *vectorizedFunction() const { return VectorizedFn; }

  /// @brief Set the vectorized function. This updates the function arguments.
  ///
  /// @param[in] NewFunction Vectorized function.
  void setVectorizedFunction(llvm::Function *NewFunction);

  /// @brief Name of the current function.
  llvm::StringRef getName() const { return function().getName(); }

  /// @brief Get the result of the vectorization
  /// @return The VectorizationResult respresenting the vectorized function
  VectorizationResult getResult() const;

  /// @brief Get the Vecz optimizations tracker class
  /// @return The Choices
  const VectorizationChoices &choices() const { return Choices; };

private:
  /// @brief Context this function is vectorized in.
  VectorizationContext &Ctx;
  /// @brief Which Vecz code generation choices are enabled and which not
  const VectorizationChoices &Choices;
  /// @brief Function to vectorize.
  llvm::Function *ScalarFn;
  /// @brief Target (vectorized) function.
  llvm::Function *VectorizedFn;
  /// @brief Arguments of the function to vectorize.
  llvm::SmallVector<VectorizerTargetArgument, 4> Arguments;
  /// @brief Vectorization factor to use.
  llvm::ElementCount SimdWidth;
  /// @brief The work group size along the vectorization dimension, if known,
  /// zero otherwise. For our purposes, this only need be an upper bound.
  uint64_t LocalSize;
  /// @brief Use the SIMD Width Analysis to determine the SIMD width
  bool AutoSimdWidth;
  /// @brief SimdDimIdx Index of vectorization dimension to use.
  unsigned SimdDimIdx;
  /// @brief Name of the builtin function, if the function to vectorize is one.
  std::string BuiltinName;
  /// @brief Per-function analysis flags.
  FunctionFlags FnFlags;
  /// @brief Placeholder instructions for arguments that will be vectorized.
  llvm::SmallPtrSet<const llvm::Instruction *, 4> ArgumentPlaceholders;
};

} // namespace vecz

#endif // VECZ_VECTORIZATION_UNIT_H_INCLUDED
