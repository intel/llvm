//===--- LoopAttributes.h - Shared loop hint state ---------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the state of the attributes and hints that may be specified
// on a loop, and the logic to determine it from the loop's statement
// attributes and the command line options. Both classic CodeGen and CIR
// CodeGen use it, and each translates the state into its own representation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_LOOPATTRIBUTES_H
#define LLVM_CLANG_CODEGENUTILS_LOOPATTRIBUTES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <optional>
#include <utility>

namespace llvm {
class MDNode;
} // namespace llvm

namespace clang {
class ASTContext;
class Attr;
class CodeGenOptions;
class ValueDecl;
} // namespace clang

namespace clang::CodeGenUtils {

/// Attributes that may be specified on loops.
struct LoopAttributes {
  explicit LoopAttributes(bool IsParallel = false);
  void clear();

  /// Generate llvm.loop.parallel metadata for loads and stores.
  bool IsParallel;

  /// State of loop vectorization or unrolling.
  enum LVEnableState { Unspecified, Enable, Disable, Full };

  /// Value for llvm.loop.vectorize.enable metadata.
  LVEnableState VectorizeEnable;

  /// Value for llvm.loop.unroll.* metadata (enable, disable, or full).
  LVEnableState UnrollEnable;

  /// Value for llvm.loop.unroll_and_jam.* metadata (enable, disable, or full).
  LVEnableState UnrollAndJamEnable;

  /// Value for llvm.loop.vectorize.predicate metadata
  LVEnableState VectorizePredicateEnable;

  /// Value for llvm.loop.vectorize.width metadata.
  unsigned VectorizeWidth;

  // Value for llvm.loop.vectorize.scalable.enable
  LVEnableState VectorizeScalable;

  /// Value for llvm.loop.interleave.count metadata.
  unsigned InterleaveCount;

  // SYCLIVDepInfo represents a group of arrays that have the same IVDep safelen to
  // them. The arrays contained in it will later be referred to from the same
  // "llvm.loop.parallel_access_indices" metadata node.
  struct SYCLIVDepInfo {
    unsigned SafeLen;
    mutable llvm::SmallVector<std::pair<const ValueDecl *, llvm::MDNode *>, 4>
        Arrays;
    SYCLIVDepInfo(unsigned SL) : SafeLen(SL) {}
    SYCLIVDepInfo(unsigned SL, const ValueDecl *A, llvm::MDNode *MD) : SafeLen(SL) {
      Arrays.emplace_back(A, MD);
    }

    bool hasArray(const ValueDecl *Array) const {
      return Arrays.end() != getArrayPairItr(Array);
    }

    decltype(Arrays)::iterator getArrayPairItr(const ValueDecl *Array) {
      return find_if(Arrays,
                     [Array](const auto &Pair) { return Pair.first == Array; });
    }

    decltype(Arrays)::iterator getArrayPairItr(const ValueDecl *Array) const {
      return find_if(Arrays,
                     [Array](const auto &Pair) { return Pair.first == Array; });
    }

    void eraseArray(const ValueDecl *Array) {
      assert(hasArray(Array) && "Precondition of EraseArray is HasArray");
      Arrays.erase(getArrayPairItr(Array));
    }

    bool isSafeLenGreaterOrEqual(unsigned OtherSL) const {
      return SafeLen == 0 || (OtherSL != 0 && SafeLen >= OtherSL);
    }
  };

  // Value for llvm.loop.parallel_access_indices metadata, for the arrays that
  // weren't put into a specific ivdep item.
  std::optional<SYCLIVDepInfo> GlobalSYCLIVDepInfo;
  // Value for llvm.loop.parallel_access_indices metadata, for array
  // specifications.
  llvm::SmallVector<SYCLIVDepInfo, 4> ArraySYCLIVDepInfo;

  /// Value for llvm.loop.ii.count metadata.
  unsigned SYCLIInterval;

  /// Value for llvm.loop.max_concurrency.count metadata.
  std::optional<unsigned> SYCLMaxConcurrencyNThreads;

  /// Flag for llvm.loop.intel.pipelining.enable, i32 0 metadata.
  bool SYCLLoopPipeliningDisable;

  /// Value for llvm.loop.max_interleaving.count metadata.
  std::optional<unsigned> SYCLMaxInterleavingNInvocations;

  /// Value for llvm.loop.intel.speculated.iterations.count metadata.
  std::optional<unsigned> SYCLSpeculatedIterationsNIterations;

  // Value for llvm.loop.intel.max_reinvocation_delay metadata.
  std::optional<unsigned> SYCLMaxReinvocationDelayNCycles;

  /// Flag for llvm.loop.intel.pipelining.enable, i32 1 metadata.
  bool SYCLLoopPipeliningEnable;

  /// llvm.unroll.
  unsigned UnrollCount;

  /// llvm.unroll.
  unsigned UnrollAndJamCount;

  /// Value for llvm.loop.distribute.enable metadata.
  LVEnableState DistributeEnable;

  /// Value for llvm.loop.pipeline.disable metadata.
  bool PipelineDisabled;

  /// Value for llvm.licm.disable metadata.
  bool LICMDisabled;

  /// Value for llvm.loop.pipeline.iicount metadata.
  unsigned PipelineInitiationInterval;

  /// Value for 'llvm.loop.align' metadata.
  unsigned CodeAlign;

  /// Value for whether the loop is required to make progress.
  bool MustProgress;
};

/// Update \p Attrs with the loop hints in \p StmtAttrs (`#pragma clang loop`,
/// `#pragma unroll`, `[[clang::code_align]]`, the OpenCL and HLSL spellings),
/// the command line options that adjust every loop (-falign-loops and
/// -fno-unroll-loops), and whether the loop must make progress.
void updateLoopAttributes(LoopAttributes &Attrs, ASTContext &Ctx,
                          const CodeGenOptions &CGOpts,
                          llvm::ArrayRef<const Attr *> StmtAttrs,
                          bool MustProgress);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_LOOPATTRIBUTES_H
