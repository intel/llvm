//===---- CGLoopInfo.h - LLVM CodeGen for loop metadata -*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal state used for llvm translation for loop statement
// metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGLOOPINFO_H
#define LLVM_CLANG_LIB_CODEGEN_CGLOOPINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class BasicBlock;
class Instruction;
class MDNode;
} // end namespace llvm

namespace clang {
class Attr;
class ASTContext;
class CodeGenOptions;
class ValueDecl;
namespace CodeGen {

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
  llvm::Optional<SYCLIVDepInfo> GlobalSYCLIVDepInfo;
  // Value for llvm.loop.parallel_access_indices metadata, for array
  // specifications.
  llvm::SmallVector<SYCLIVDepInfo, 4> ArraySYCLIVDepInfo;

  /// Value for llvm.loop.ii.count metadata.
  unsigned SYCLIInterval;

  /// Value for llvm.loop.max_concurrency.count metadata.
  llvm::Optional<unsigned> SYCLMaxConcurrencyNThreads;

  /// Value for count variant (min/max/avg) and count metadata.
  llvm::SmallVector<std::pair<const char *, unsigned int>, 2>
      SYCLIntelFPGAVariantCount;

  /// Flag for llvm.loop.coalesce metadata.
  bool SYCLLoopCoalesceEnable;

  /// Value for llvm.loop.coalesce.count metadata.
  unsigned SYCLLoopCoalesceNLevels;

  /// Flag for llvm.loop.intel.pipelining.enable, i32 0 metadata.
  bool SYCLLoopPipeliningDisable;

  /// Value for llvm.loop.max_interleaving.count metadata.
  llvm::Optional<unsigned> SYCLMaxInterleavingNInvocations;

  /// Value for llvm.loop.intel.speculated.iterations.count metadata.
  llvm::Optional<unsigned> SYCLSpeculatedIterationsNIterations;

  /// llvm.unroll.
  unsigned UnrollCount;

  /// llvm.unroll.
  unsigned UnrollAndJamCount;

  /// Value for llvm.loop.distribute.enable metadata.
  LVEnableState DistributeEnable;

  /// Value for llvm.loop.pipeline.disable metadata.
  bool PipelineDisabled;

  /// Value for llvm.loop.pipeline.iicount metadata.
  unsigned PipelineInitiationInterval;

  /// Flag for llvm.loop.fusion.disable metatdata.
  bool SYCLNofusionEnable;

  /// Value for whether the loop is required to make progress.
  bool MustProgress;
};

/// Information used when generating a structured loop.
class LoopInfo {
public:
  /// Construct a new LoopInfo for the loop with entry Header.
  LoopInfo(llvm::BasicBlock *Header, const LoopAttributes &Attrs,
           const llvm::DebugLoc &StartLoc, const llvm::DebugLoc &EndLoc,
           LoopInfo *Parent);

  /// Get the loop id metadata for this loop.
  llvm::MDNode *getLoopID() const { return TempLoopID.get(); }

  /// Get the header block of this loop.
  llvm::BasicBlock *getHeader() const { return Header; }

  /// Get the set of attributes active for this loop.
  const LoopAttributes &getAttributes() const { return Attrs; }

  /// Return this loop's access group or nullptr if it does not have one.
  llvm::MDNode *getAccessGroup() const { return AccGroup; }

  // Recursively adds the metadata for this Array onto this GEP.
  void addIVDepMetadata(const ValueDecl *Array, llvm::Instruction *GEP) const;

  /// Create the loop's metadata. Must be called after its nested loops have
  /// been processed.
  void finish();

private:
  /// Loop ID metadata.
  llvm::TempMDTuple TempLoopID;
  /// Header block of this loop.
  llvm::BasicBlock *Header;
  /// The attributes for this loop.
  LoopAttributes Attrs;
  /// The access group for memory accesses parallel to this loop.
  llvm::MDNode *AccGroup = nullptr;
  /// Start location of this loop.
  llvm::DebugLoc StartLoc;
  /// End location of this loop.
  llvm::DebugLoc EndLoc;
  /// The next outer loop, or nullptr if this is the outermost loop.
  LoopInfo *Parent;
  /// If this loop has unroll-and-jam metadata, this can be set by the inner
  /// loop's LoopInfo to set the llvm.loop.unroll_and_jam.followup_inner
  /// metadata.
  llvm::MDNode *UnrollAndJamInnerFollowup = nullptr;

  /// Create a LoopID without any transformations.
  llvm::MDNode *
  createLoopPropertiesMetadata(llvm::ArrayRef<llvm::Metadata *> LoopProperties);

  /// Create a LoopID for transformations.
  ///
  /// The methods call each other in case multiple transformations are applied
  /// to a loop. The transformation first to be applied will use LoopID of the
  /// next transformation in its followup attribute.
  ///
  /// @param Attrs             The loop's transformations.
  /// @param LoopProperties    Non-transformation properties such as debug
  ///                          location, parallel accesses and disabled
  ///                          transformations. These are added to the returned
  ///                          LoopID.
  /// @param HasUserTransforms [out] Set to true if the returned MDNode encodes
  ///                          at least one transformation.
  ///
  /// @return A LoopID (metadata node) that can be used for the llvm.loop
  ///         annotation or followup-attribute.
  /// @{
  llvm::MDNode *
  createPipeliningMetadata(const LoopAttributes &Attrs,
                           llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                           bool &HasUserTransforms);
  llvm::MDNode *
  createPartialUnrollMetadata(const LoopAttributes &Attrs,
                              llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                              bool &HasUserTransforms);
  llvm::MDNode *
  createUnrollAndJamMetadata(const LoopAttributes &Attrs,
                             llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                             bool &HasUserTransforms);
  llvm::MDNode *
  createLoopVectorizeMetadata(const LoopAttributes &Attrs,
                              llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                              bool &HasUserTransforms);
  llvm::MDNode *
  createLoopDistributeMetadata(const LoopAttributes &Attrs,
                               llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                               bool &HasUserTransforms);
  llvm::MDNode *
  createFullUnrollMetadata(const LoopAttributes &Attrs,
                           llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                           bool &HasUserTransforms);
  void collectIVDepMetadata(const ValueDecl *Array,
                            llvm::SmallVectorImpl<llvm::Metadata *> &MD) const;
  /// @}

  /// Create a LoopID for this loop, including transformation-unspecific
  /// metadata such as debug location.
  ///
  /// @param Attrs             This loop's attributes and transformations.
  /// @param LoopProperties    Additional non-transformation properties to add
  ///                          to the LoopID, such as transformation-specific
  ///                          metadata that are not covered by @p Attrs.
  /// @param HasUserTransforms [out] Set to true if the returned MDNode encodes
  ///                          at least one transformation.
  ///
  /// @return A LoopID (metadata node) that can be used for the llvm.loop
  ///         annotation.
  llvm::MDNode *createMetadata(const LoopAttributes &Attrs,
                               llvm::ArrayRef<llvm::Metadata *> LoopProperties,
                               bool &HasUserTransforms);
};

/// A stack of loop information corresponding to loop nesting levels.
/// This stack can be used to prepare attributes which are applied when a loop
/// is emitted.
class LoopInfoStack {
  LoopInfoStack(const LoopInfoStack &) = delete;
  void operator=(const LoopInfoStack &) = delete;

public:
  LoopInfoStack() {}

  /// Begin a new structured loop. The set of staged attributes will be
  /// applied to the loop and then cleared.
  void push(llvm::BasicBlock *Header, const llvm::DebugLoc &StartLoc,
            const llvm::DebugLoc &EndLoc);

  /// Begin a new structured loop. Stage attributes from the Attrs list.
  /// The staged attributes are applied to the loop and then cleared.
  void push(llvm::BasicBlock *Header, clang::ASTContext &Ctx,
            const clang::CodeGenOptions &CGOpts,
            llvm::ArrayRef<const Attr *> Attrs, const llvm::DebugLoc &StartLoc,
            const llvm::DebugLoc &EndLoc, bool MustProgress = false);

  /// End the current loop.
  void pop();

  /// Return the top loop id metadata.
  llvm::MDNode *getCurLoopID() const { return getInfo().getLoopID(); }

  /// Return true if the top loop is parallel.
  bool getCurLoopParallel() const {
    return hasInfo() ? getInfo().getAttributes().IsParallel : false;
  }

  /// Function called by the CodeGenFunction when an instruction is
  /// created.
  void InsertHelper(llvm::Instruction *I) const;

  /// Set the next pushed loop as parallel.
  void setParallel(bool Enable = true) { StagedAttrs.IsParallel = Enable; }

  /// Set the next pushed loop 'vectorize.enable'
  void setVectorizeEnable(bool Enable = true) {
    StagedAttrs.VectorizeEnable =
        Enable ? LoopAttributes::Enable : LoopAttributes::Disable;
  }

  /// Set the next pushed loop as a distribution candidate.
  void setDistributeState(bool Enable = true) {
    StagedAttrs.DistributeEnable =
        Enable ? LoopAttributes::Enable : LoopAttributes::Disable;
  }

  /// Set the next pushed loop unroll state.
  void setUnrollState(const LoopAttributes::LVEnableState &State) {
    StagedAttrs.UnrollEnable = State;
  }

  /// Set the next pushed vectorize predicate state.
  void setVectorizePredicateState(const LoopAttributes::LVEnableState &State) {
    StagedAttrs.VectorizePredicateEnable = State;
  }

  /// Set the next pushed loop unroll_and_jam state.
  void setUnrollAndJamState(const LoopAttributes::LVEnableState &State) {
    StagedAttrs.UnrollAndJamEnable = State;
  }

  /// Set the vectorize width for the next loop pushed.
  void setVectorizeWidth(unsigned W) { StagedAttrs.VectorizeWidth = W; }

  void setVectorizeScalable(const LoopAttributes::LVEnableState &State) {
    StagedAttrs.VectorizeScalable = State;
  }

  /// Set the interleave count for the next loop pushed.
  void setInterleaveCount(unsigned C) { StagedAttrs.InterleaveCount = C; }

  /// Add a safelen value for the next loop pushed.
  void addSYCLIVDepInfo(llvm::LLVMContext &Ctx, unsigned SafeLen,
                        const ValueDecl *Array);

  void addIVDepMetadata(const ValueDecl *Array, llvm::Instruction *GEP);

  /// Set value of an initiation interval for the next loop pushed.
  void setSYCLIInterval(unsigned C) { StagedAttrs.SYCLIInterval = C; }

  /// Set value of max_concurrency for the next loop pushed.
  void setSYCLMaxConcurrencyNThreads(unsigned C) {
    StagedAttrs.SYCLMaxConcurrencyNThreads = C;
  }

  /// Set flag of loop_coalesce for the next loop pushed.
  void setSYCLLoopCoalesceEnable() {
    StagedAttrs.SYCLLoopCoalesceEnable = true;
  }

  /// Set value of coalesced levels for the next loop pushed.
  void setSYCLLoopCoalesceNLevels(unsigned C) {
    StagedAttrs.SYCLLoopCoalesceNLevels = C;
  }

  /// Set flag of disable_loop_pipelining for the next loop pushed.
  void setSYCLLoopPipeliningDisable() {
    StagedAttrs.SYCLLoopPipeliningDisable = true;
  }

  /// Set value of max interleaved invocations for the next loop pushed.
  void setSYCLMaxInterleavingNInvocations(unsigned C) {
    StagedAttrs.SYCLMaxInterleavingNInvocations = C;
  }

  /// Set value of speculated iterations for the next loop pushed.
  void setSYCLSpeculatedIterationsNIterations(unsigned C) {
    StagedAttrs.SYCLSpeculatedIterationsNIterations = C;
  }

  /// Set value of variant and loop count for the next loop pushed.
  void setSYCLIntelFPGAVariantCount(const char *Var, unsigned int Count) {
    StagedAttrs.SYCLIntelFPGAVariantCount.push_back({Var, Count});
  }

  /// Set the unroll count for the next loop pushed.
  void setUnrollCount(unsigned C) { StagedAttrs.UnrollCount = C; }

  /// \brief Set the unroll count for the next loop pushed.
  void setUnrollAndJamCount(unsigned C) { StagedAttrs.UnrollAndJamCount = C; }

  /// Set the pipeline disabled state.
  void setPipelineDisabled(bool S) { StagedAttrs.PipelineDisabled = S; }

  /// Set the pipeline initiation interval.
  void setPipelineInitiationInterval(unsigned C) {
    StagedAttrs.PipelineInitiationInterval = C;
  }

  /// Set flag of nofusion for the next loop pushed.
  void setSYCLNofusionEnable() { StagedAttrs.SYCLNofusionEnable = true; }

  /// Set no progress for the next loop pushed.
  void setMustProgress(bool P) { StagedAttrs.MustProgress = P; }

private:
  /// Returns true if there is LoopInfo on the stack.
  bool hasInfo() const { return !Active.empty(); }
  /// Return the LoopInfo for the current loop. HasInfo should be called
  /// first to ensure LoopInfo is present.
  const LoopInfo &getInfo() const { return *Active.back(); }
  /// The set of attributes that will be applied to the next pushed loop.
  LoopAttributes StagedAttrs;
  /// Stack of active loops.
  llvm::SmallVector<std::unique_ptr<LoopInfo>, 4> Active;
};

} // end namespace CodeGen
} // end namespace clang

#endif
