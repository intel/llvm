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

/// @file vectorization_context.h
///
/// @brief Hold global state and objects used for vectorization.

#ifndef VECZ_VECTORIZATION_CONTEXT_H_INCLUDED
#define VECZ_VECTORIZATION_CONTEXT_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/Support/AtomicOrdering.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <multi_llvm/multi_llvm.h>

#include <map>
#include <memory>

namespace llvm {
class TargetTransformInfo;
}  // namespace llvm

namespace compiler {
namespace utils {
class BuiltinInfo;
}  // namespace utils
}  // namespace compiler

namespace vecz {
class MemOpDesc;
class TargetInfo;
struct UniformValueResult;
class VectorizationChoices;
struct VectorizationResult;
class VectorizationUnit;

using ActiveUnitMap = llvm::DenseMap<llvm::PoisoningVH<const llvm::Function>,
                                     VectorizationUnit *>;

/// @brief Holds global (per-module) vectorization state.
class VectorizationContext {
 public:
  /// @brief Create a new vectorization context object.
  ///
  /// @param[in] target Module in which vectorization happens.
  /// @param[in] vti Target information.
  /// @param[in] bi Builtins information
  VectorizationContext(llvm::Module &target, TargetInfo &vti,
                       compiler::utils::BuiltinInfo &bi);

  /// @brief Access the public vectorizer API.

  /// @brief Module in which vectorization happens.
  llvm::Module &module() const { return Module; }

  /// @brief Data layout for the target.
  const llvm::DataLayout *dataLayout() const { return DL; }

  /// @brief Information about the target.
  TargetInfo &targetInfo() { return VTI; }

  /// @brief Information about the target.
  const TargetInfo &targetInfo() const { return VTI; }

  llvm::TargetTransformInfo getTargetTransformInfo(llvm::Function &F) const;

  /// @brief Construct and initialize the PassManager to be used for
  /// vectorizing.
  /// @return true if no problem occurred, false otherwise.
  bool buildPassPipeline();
  VectorizationUnit *getActiveVU(const llvm::Function *F) const;

  /// @brief Log the Function's VectorizationUnit as the one governing the
  /// current vectorization.
  void setActiveVU(llvm::Function *F, VectorizationUnit *VU) {
    ActiveVUs[F] = VU;
  }
  /// @brief Log the Function's VectorizationUnit as the one governing the
  /// current vectorization.
  void clearActiveVU(llvm::Function *F) { ActiveVUs.erase(F); }

  /// @brief Builtin database.
  compiler::utils::BuiltinInfo &builtins();

  /// @brief Builtin database.
  const compiler::utils::BuiltinInfo &builtins() const;

  /// @brief Determine whether the function is an internal builtin or not.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return true if F is an internal builtin function, false otherwise.
  static bool isInternalBuiltin(const llvm::Function *F);
  /// @brief Create a new function with the given name and type, unless it
  /// already exists in the module. Mark it as an internal builtin.
  ///
  /// @param[in] Name Name of the builtin function.
  /// @param[in] FT Function type for the builtin.
  ///
  /// @return Internal builtin function with the given Name.
  llvm::Function *getOrCreateInternalBuiltin(llvm::StringRef Name,
                                             llvm::FunctionType *FT = nullptr);
  /// @brief Define the internal builtin function, i.e. generate its body.
  ///
  /// @param[in] F Function declaration to emit a body for.
  ///
  /// @return true if the body of the builtin was emitted, false otherwise.
  bool defineInternalBuiltin(llvm::Function *F);
  /// @brief Given a scalar builtin function, return a vector equivalent if it
  /// is an internal builtin.
  ///
  /// @param[in] ScalarFn Scalar builtin to map to a vector equivalent.
  /// @param[in] SimdWidth SIMD width used to determine which vector equivalent
  /// to select.
  ///
  /// @return Equivalent vector builtin function on success, or null.
  llvm::Function *getInternalVectorEquivalent(llvm::Function *ScalarFn,
                                              unsigned SimdWidth);

  /// @brief Check if the given function is a masked version of another function
  ///
  /// @param[in] F The function to check
  /// @return true if the function is a masked version, or false otherwise
  bool isMaskedFunction(const llvm::Function *F) const;
  /// @brief Get the original non-masked function from a masked function
  ///
  /// @param[in] F The masked function
  /// @return Original masked function if it exists, or null
  llvm::Function *getOriginalMaskedFunction(llvm::Function *F);
  /// @brief Get (if it exists already) or create the masked version of a
  /// function
  ///
  /// @param[in] CI Call to the function to be masked
  /// @return The masked version of the function
  llvm::Function *getOrCreateMaskedFunction(llvm::CallInst *CI);

  /// @brief Represents either an atomicrmw or cmpxchg operation.
  ///
  /// Most fields are shared, with the exception of CmpXchgFailureOrdering and
  /// IsWeak, which are only to be set for cmpxchg, and BinOp, which is only to
  /// be set to a valid value for atomicrmw.
  struct MaskedAtomic {
    llvm::Type *PointerTy;
    llvm::Type *ValTy;
    /// @brief Must be set to BAD_BINOP for cmpxchg instructions
    llvm::AtomicRMWInst::BinOp BinOp;
    llvm::Align Align;
    bool IsVolatile = false;
    llvm::SyncScope::ID SyncScope;
    llvm::AtomicOrdering Ordering;
    /// @brief Must be set for cmpxchg instructions
    std::optional<llvm::AtomicOrdering> CmpXchgFailureOrdering = std::nullopt;
    /// @brief Must only be set for cmpxchg instructions
    bool IsWeak = false;
    // Vectorization info
    llvm::ElementCount VF;
    bool IsVectorPredicated = false;

    /// @brief Returns true if this MaskedAtomic represents a cmpxchg operation.
    bool isCmpXchg() const {
      if (CmpXchgFailureOrdering.has_value()) {
        // 'binop' only applies to atomicrmw
        assert(BinOp == llvm::AtomicRMWInst::BAD_BINOP &&
               "Invalid MaskedAtomic state");
        return true;
      }
      // 'weak' only applies to cmpxchg
      assert(!IsWeak && "Invalid MaskedAtomic state");
      return false;
    }
  };

  /// @brief Check if the given function is a masked version of an atomicrmw or
  /// cmpxchg operation.
  ///
  /// @param[in] F The function to check
  /// @return A MaskedAtomic instance detailing the atomic operation if the
  /// function is a masked atomic, or std::nullopt otherwise
  std::optional<MaskedAtomic> isMaskedAtomicFunction(
      const llvm::Function &F) const;
  /// @brief Get (if it exists already) or create the function representing the
  /// masked version of an atomicrmw/cmpxchg operation.
  ///
  /// @param[in] I Atomic to be masked
  /// @param[in] Choices Choices to mangle into the function name
  /// @param[in] VF The vectorization factor of the atomic operation
  /// @return The masked version of the function
  llvm::Function *getOrCreateMaskedAtomicFunction(
      MaskedAtomic &I, const VectorizationChoices &Choices,
      llvm::ElementCount VF);

  /// @brief Create a VectorizationUnit to use to vectorize the given scalar
  /// function.
  ///
  /// The lifetime of the returned VectorizationUnit is managed by the
  /// VectorizationContext.
  ///
  /// @param[in] F Function to vectorize.
  /// @param[in] VF vectorization factor to use.
  /// @param[in] Dimension SIMD dimension to use (0 => x, 1 => y, 2 => z).
  /// @param[in] Ch Vectorization Choices for the vectorization.
  VectorizationUnit *createVectorizationUnit(llvm::Function &F,
                                             llvm::ElementCount VF,
                                             unsigned Dimension,
                                             const VectorizationChoices &Ch);

  /// @brief Vectorizes all Vectorization Units in the context
  void vectorize();

  /// @brief Try to get a vectorization result for the scalar builtin function.
  ///
  /// @param[in] F Builtin function to create or retrieve an unit for.
  /// @param[in] SimdWidth Vectorization factor to use.
  ///
  /// @return a VectorizationResult representing the vectorized function.
  VectorizationResult &getOrCreateBuiltin(llvm::Function &F,
                                          unsigned SimdWidth);

  /// @brief Vectorize a builtin function by a given factor
  ///
  /// @param[in] F the function to vectorize.
  /// @param[in] factor the vectorization factor.
  ///
  /// @return a VectorizationResult representing the vectorized function.
  VectorizationResult getVectorizedFunction(llvm::Function &F,
                                            llvm::ElementCount factor);

  /// @brief Determine whether I is a vector instruction or not, i.e. it has any
  /// vector operand.
  ///
  /// @param[in] I Instruction to analyze.
  ///
  /// @return true if I is a vector instruction.
  static bool isVector(const llvm::Instruction &I);

  static const char *InternalBuiltinPrefix;

 private:
  /// @brief Determine whether this scalar builtin function can be safely
  /// expanded at vector call sites, i.e. it has not side effects.
  ///
  /// @param[in] ScalarFn Builtin function to analyze.
  ///
  /// @return true if the function can be expanded.
  bool canExpandBuiltin(const llvm::Function *ScalarFn) const;

  /// @brief Emit the body for the masked load or store internal builtins
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] Desc The MemOpDesc for the memory operation
  /// @returns true on success, false otherwise
  bool emitMaskedMemOpBody(llvm::Function &F, const MemOpDesc &Desc) const;
  /// @brief Emit the body for the interleaved load or store internal builtins
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] Desc The MemOpDesc for the memory operation
  /// @returns true on success, false otherwise
  bool emitInterleavedMemOpBody(llvm::Function &F, const MemOpDesc &Desc) const;
  /// @brief Emit the body for the masked interleaved load/store internal
  /// builtins
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] Desc The MemOpDesc for the memory operation
  /// @returns true on success, false otherwise
  bool emitMaskedInterleavedMemOpBody(llvm::Function &F,
                                      const MemOpDesc &Desc) const;
  /// @brief Emit the body for the scatter or gather internal builtins
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] Desc The MemOpDesc for the memory operation
  /// @returns true on success, false otherwise
  bool emitScatterGatherMemOpBody(llvm::Function &F,
                                  const MemOpDesc &Desc) const;
  /// @brief Emit the body for the masked scatter or gather internal builtins
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] Desc The MemOpDesc for the memory operation
  /// @returns true on success, false otherwise
  bool emitMaskedScatterGatherMemOpBody(llvm::Function &F,
                                        const MemOpDesc &Desc) const;
  /// @brief Add the masked function to the tracking set
  ///
  /// @param[in] F The function to add
  /// @param[in] WrappedF The original function being masked
  /// @return false if the function was already in the set, or true otherwise
  bool insertMaskedFunction(llvm::Function *F, llvm::Function *WrappedF);

  /// @brief Emit the body for the subgroup scan builtins
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] IsInclusive whether the scan should be inclusive (on true) or
  /// exclusive (on false).
  /// @param[in] OpKind the kind of scan to emit. Note: not all values of
  /// llvm::RecurKind are supported scan operations.
  /// @param[in] IsVP whether the scan is vector-predicated.
  /// @returns true on success, false otherwise
  bool emitSubgroupScanBody(llvm::Function &F, bool IsInclusive,
                            llvm::RecurKind OpKind, bool IsVP) const;

  /// @brief Emit the body for a masked atomic builtin
  ///
  /// @param[in] F The empty (declaration only) function to emit the body in
  /// @param[in] MA The MaskedAtomic information
  /// @returns true on success, false otherwise
  bool emitMaskedAtomicBody(llvm::Function &F, const MaskedAtomic &MA) const;

  /// @brief Helper for non-vectorization tasks.
  TargetInfo &VTI;
  /// @brief Module in which the vectorization happens.
  llvm::Module &Module;
  /// @brief Builtins database.
  compiler::utils::BuiltinInfo &BI;
  /// @brief Data layout object used to determine the size and alignment of
  /// types.
  const llvm::DataLayout *DL;
  /// @brief Persistent storage for Kernel Vectorization Units
  std::vector<std::unique_ptr<VectorizationUnit>> KernelUnits;
  /// @brief Mapping between functions in the module and vectorization units.
  llvm::DenseMap<const llvm::Function *,
                 llvm::SmallDenseMap<unsigned, VectorizationResult, 1>>
      VectorizedBuiltins;
  /// @brief Maps vector functions to their VectorizationUnits
  ActiveUnitMap ActiveVUs;
  /// @brief Map of masked functions used in the module to their original
  /// non-masked function.
  llvm::ValueToValueMapTy MaskedFunctionsMap;
  /// @brief All the masked versions of functions generated by Vecz
  ///
  /// Keeps track of all the functions we already have masked versions of. We
  /// use the name of the masked function instead of just the Function pointer
  /// because vararg functions have different masked versions for different
  /// argument types.
  std::map<std::string, llvm::Function *> MaskedVersions;
};

/// \addtogroup passes Passes
/// @{
/// \ingroup vecz

/// @brief Implement internal builtins.
class DefineInternalBuiltinsPass
    : public llvm::PassInfoMixin<DefineInternalBuiltinsPass> {
 public:
  /// @brief Create a new pass object.
  DefineInternalBuiltinsPass() {}

  static void *ID() { return (void *)&PassID; }

  /// @brief Define all used internal builtins in the module, expanding bodies
  /// for declaration only references.
  ///
  /// @param[in] M Module in which to define internal builtins.
  /// @param[in] AM ModuleAnalysisManager providing analyses.
  ///
  /// @return Set of preserved analyses (all analyses).
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  static llvm::StringRef name() { return "Define internal builtins"; }

 private:
  /// @brief Identifier for the DefineInternalBuiltin pass.
  static char PassID;
};

/// @}
}  // namespace vecz

#endif  // VECZ_VECTORIZATION_CONTEXT_H_INCLUDED
