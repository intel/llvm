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
/// @brief External vecz header.  Contains the API to the vectorizer.

#ifndef VECZ_VECZ_TARGET_INFO_H_INCLUDED
#define VECZ_VECZ_TARGET_INFO_H_INCLUDED

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/TypeSize.h>

namespace llvm {
class TargetMachine;
class TargetTransformInfo;
class Type;
}  // namespace llvm

namespace vecz {
class VectorizationContext;

/// @addtogroup vecz
/// @{

/// @brief Kinds of interleaved memory operations.
enum InterleavedOperation : int {
  /// @brief Invalid memory operation.
  eInterleavedInvalid = 0,
  /// @brief Store memory operation.
  eInterleavedStore,
  /// @brief Load memory operation.
  eInterleavedLoad,
  /// @brief Masked Store memory operation.
  eMaskedInterleavedStore,
  /// @brief Masked Load memory operation.
  eMaskedInterleavedLoad
};

/// @brief Used by the vectorizer to query for target capabilities and
/// materialize memory intrinsics.
class TargetInfo {
 public:
  /// @brief Create a new vector target info instance.
  /// @param[in] tm LLVM target machine that will be used for compilation, can
  /// be NULL if no target data is available.
  TargetInfo(llvm::TargetMachine *tm);

  virtual ~TargetInfo() = default;

  /// @brief Return the target machine.
  llvm::TargetMachine *getTargetMachine() const { return TM_; }

  /// @brief Create a vector load. If a stride greater than one is used, the
  /// load will be interleaved, i.e. lanes are loaded from non-contiguous
  /// memory.
  ///
  /// @note ptr refers to the unwidened element type, not the wide type.
  ///       ptr needs to be 'element aligned'. The element can itself be a
  ///       vector.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Value type to load from memory.
  /// @param[in] ptr Memory address to load a vector value from.
  /// @param[in] stride Distance in elements between two lanes in memory.
  ///                     A stride of one represents a contiguous load.
  /// @param[in] alignment The alignment of the load, in bytes
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If null, the operation is unpredicated:
  /// it is executed on all lanes.
  ///
  /// @return IR value that results from the vector load.
  virtual llvm::Value *createLoad(llvm::IRBuilder<> &builder, llvm::Type *ty,
                                  llvm::Value *ptr, llvm::Value *stride,
                                  unsigned alignment,
                                  llvm::Value *evl = nullptr) const;

  /// @brief Create a vector store. If a stride greater than one is used, the
  /// store will be interleaved, i.e. lanes are stored to non-contiguous memory.
  ///
  /// @note ptr refers to the unwidened element type, not the wide type.
  ///       ptr needs to be 'element aligned'. The element can itself be a
  ///       vector.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] data Vector value to store to memory.
  /// @param[in] ptr Memory address to store a vector value to.
  /// @param[in] stride Distance in elements between two lanes in memory.
  ///                     A stride of one represents a contiguous store.
  /// @param[in] alignment The alignment of the store, in bytes
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If null, the operation is unpredicated:
  /// it is executed on all lanes.
  ///
  /// @return IR value that results from the vector store.
  virtual llvm::Value *createStore(llvm::IRBuilder<> &builder,
                                   llvm::Value *data, llvm::Value *ptr,
                                   llvm::Value *stride, unsigned alignment,
                                   llvm::Value *evl = nullptr) const;

  /// @brief Create a masked vector load.
  ///        Only lanes with a non-zero mask will be loaded from the address.
  ///        Other lanes will contain undefined data.
  ///
  /// @note ptr refers to the unwidened element type, not the wide type.
  ///       ptr needs to be 'element aligned'. The element can itself be a
  ///       vector.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Value type to load from memory.
  /// @param[in] ptr Memory address to load a vector value from.
  /// @param[in] mask Vector mask used to disable loading certain lanes.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the store.
  ///
  /// @return IR value that results from the masked vector load.
  virtual llvm::Value *createMaskedLoad(llvm::IRBuilder<> &builder,
                                        llvm::Type *ty, llvm::Value *ptr,
                                        llvm::Value *mask, llvm::Value *evl,
                                        unsigned alignment) const;

  /// @brief Create a masked vector store.
  ///        Only lanes with a non-zero mask will be stored to the address.
  ///
  /// @note ptr refers to the unwidened element type, not the wide type.
  ///       ptr needs to be 'element aligned'. The element can itself be a
  ///       vector.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] data Vector value to store to memory.
  /// @param[in] ptr Memory address to store a vector value to.
  /// @param[in] mask Vector mask used to disable storing certain lanes.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the store.
  ///
  /// @return IR value that results from the masked vector store.
  virtual llvm::Value *createMaskedStore(llvm::IRBuilder<> &builder,
                                         llvm::Value *data, llvm::Value *ptr,
                                         llvm::Value *mask, llvm::Value *evl,
                                         unsigned alignment) const;

  /// @brief Create a interleaved vector load.
  ///
  /// @note Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Value type to load from memory.
  /// @param[in] ptr Memory address to load a vector value to.
  /// @param[in] stride Stride for interleaved memory operation.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment  Alignment of the load
  ///
  /// @return IR value that results from the interleaved load.
  virtual llvm::Value *createInterleavedLoad(llvm::IRBuilder<> &builder,
                                             llvm::Type *ty, llvm::Value *ptr,
                                             llvm::Value *stride,
                                             llvm::Value *evl,
                                             unsigned alignment) const;

  /// @brief Create a interleaved vector store.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] data Vector value to store to memory.
  /// @param[in] ptr Memory address to store a vector value to.
  /// @param[in] stride Stride for interleaved memory operation.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the load
  ///
  /// @return IR value that results from the interleaved vector store.
  virtual llvm::Value *createInterleavedStore(
      llvm::IRBuilder<> &builder, llvm::Value *data, llvm::Value *ptr,
      llvm::Value *stride, llvm::Value *evl, unsigned alignment) const;

  /// @brief Create a masked interleaved vector load.
  ///        Only lanes with a non-zero mask will be loaded from the address.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Value type to load from memory.
  /// @param[in] ptr Memory address to load a vector value to.
  /// @param[in] mask Vector mask used to disable loading certain lanes.
  /// @param[in] stride Stride for interleaved memory operation.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the load
  ///
  /// @return IR value that results from the masked interleaved vector load.
  virtual llvm::Value *createMaskedInterleavedLoad(
      llvm::IRBuilder<> &builder, llvm::Type *ty, llvm::Value *ptr,
      llvm::Value *mask, llvm::Value *stride, llvm::Value *evl,
      unsigned alignment) const;

  /// @brief Create a masked interleaved vector store.
  ///        Only lanes with a non-zero mask will be stored to the address.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] data Vector value to store to memory.
  /// @param[in] ptr Memory address to store a vector value to.
  /// @param[in] mask Vector mask used to disable storing certain lanes.
  /// @param[in] stride Stride for interleaved memory operation.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the load
  ///
  /// @return IR value that results from the masked interleaved vector store.
  virtual llvm::Value *createMaskedInterleavedStore(
      llvm::IRBuilder<> &builder, llvm::Value *data, llvm::Value *ptr,
      llvm::Value *mask, llvm::Value *stride, llvm::Value *evl,
      unsigned alignment) const;

  /// @brief Create a gather vector load.
  ///        Vector lanes are loaded from different memory addresses.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Value type to load from memory.
  /// @param[in] ptr Memory address to load a vector value from.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the store.
  ///
  /// @return IR value that results from the gather vector load.
  virtual llvm::Value *createGatherLoad(llvm::IRBuilder<> &builder,
                                        llvm::Type *ty, llvm::Value *ptr,
                                        llvm::Value *evl,
                                        unsigned alignment) const;

  /// @brief Create a scatter vector store.
  ///        Vector lanes are stored to different memory addresses.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] data Vector value to store to memory.
  /// @param[in] ptr Memory address to store a vector value to.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the store.
  ///
  /// @return IR value that results from the scatter vector store.
  virtual llvm::Value *createScatterStore(llvm::IRBuilder<> &builder,
                                          llvm::Value *data, llvm::Value *ptr,
                                          llvm::Value *evl,
                                          unsigned alignment) const;

  /// @brief Create a masked gather vector load.
  ///        Only lanes with a non-zero mask will be loaded from different
  ///        address.
  ///        Other lanes will contain undefined data.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Value type to load from memory.
  /// @param[in] ptr Memory address to load a vector value from.
  /// @param[in] mask Vector mask used to disable loading certain lanes.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the store.
  ///
  /// @return IR value that results from the masked gather vector load.
  virtual llvm::Value *createMaskedGatherLoad(llvm::IRBuilder<> &builder,
                                              llvm::Type *ty, llvm::Value *ptr,
                                              llvm::Value *mask,
                                              llvm::Value *evl,
                                              unsigned alignment) const;

  /// @brief Create a masked scatter vector store.
  ///        Only lanes with a non-zero mask will be stored to the address.
  ///
  /// @note  Pointers are scalar and need to be 'scalar aligned'.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] data Vector value to store to memory.
  /// @param[in] ptr Memory address to store a vector value to.
  /// @param[in] mask Vector mask used to disable storing certain lanes.
  /// @param[in] evl 'effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  /// @param[in] alignment Alignment of the store.
  ///
  /// @return IR value that results from the masked scatter vector store.
  virtual llvm::Value *createMaskedScatterStore(
      llvm::IRBuilder<> &builder, llvm::Value *data, llvm::Value *ptr,
      llvm::Value *mask, llvm::Value *evl, unsigned alignment) const;

  /// @brief Create a scalable extractelement instruction. Note that the
  /// operands are expected to have been pre-packetized before passing to this
  /// function.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] Ctx Vectorization context.
  /// @param[in] extract The original pre-packetized extractelement Instruction
  /// @param[in] narrowTy Narrowed type of @a extract.
  /// @param[in] src The packetized source vector
  /// @param[in] index The packetized extraction index
  /// @param[in] evl 'Effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  ///
  /// @return A value identical to the requested extractelement
  virtual llvm::Value *createScalableExtractElement(
      llvm::IRBuilder<> &builder, vecz::VectorizationContext &Ctx,
      llvm::Instruction *extract, llvm::Type *narrowTy, llvm::Value *src,
      llvm::Value *index, llvm::Value *evl) const;

  /// @brief Create an outer broadcast of a vector. An outer broadcast is one
  /// where a vector with length V is replicated in its entirety N times across
  /// the lanes of a larger vector with length L x V. The broadcast factor is
  /// expected to be scalable:
  ///
  ///   outer_broadcast(<A,B>, vscale x 1) -> <A,B,A,B,A,B,...>
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] vector Vector to broadcast.
  /// @param[in] VL Vector length.
  /// @param[in] factor Broadcast factor.
  virtual llvm::Value *createOuterScalableBroadcast(
      llvm::IRBuilder<> &builder, llvm::Value *vector, llvm::Value *VL,
      llvm::ElementCount factor) const;

  /// @brief Create an inner broadcast of a vector. An inner broadcast is one
  /// where a vector with length V has its lanes individually and sequentially
  /// replicated N times to fill a larger vector with length L x V. The
  /// broadcast factor is expected to be a fixed amount:
  ///
  ///   inner_broadcast(<A,B,C,...>, 2) -> <A,A,B,B,C,C, ...>
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] vector Vector to broadcast.
  /// @param[in] VL Vector length.
  /// @param[in] factor Broadcast factor.
  virtual llvm::Value *createInnerScalableBroadcast(
      llvm::IRBuilder<> &builder, llvm::Value *vector, llvm::Value *VL,
      llvm::ElementCount factor) const;

  /// @brief Utility function for packetizing an insertelement instruction by a
  /// scalable factor. Note that the operands are expected to have been
  /// pre-packetized before passing to this function.
  ///
  /// @param[in] builder the builder to create the needed instructions
  /// @param[in] Ctx Vectorization context.
  /// @param[in] insert the original pre-packetized insertelement Instruction
  /// @param[in] elt the packetized element to insert
  /// @param[in] into the packetized source vector
  /// @param[in] index the packetized insertion index
  /// @param[in] evl 'Effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes but obeys the mask parameter.
  ///
  /// @return a value identical to the requested insertelement
  virtual llvm::Value *createScalableInsertElement(
      llvm::IRBuilder<> &builder, vecz::VectorizationContext &Ctx,
      llvm::Instruction *insert, llvm::Value *elt, llvm::Value *into,
      llvm::Value *index, llvm::Value *evl) const;

  /// @brief Function allowing targets to customize the insertion of
  /// instructions to calculate the vector-predicated kernel width.
  ///
  /// Note that this must return an expression equivalent to:
  ///   i32 = umin(%factor, %remainingIters)
  /// This is the expression computed if this function returns nullptr.
  ///
  /// @param[in] builder the builder to create the needed instructions
  /// @param[in] remainingIters the remaining number of work-items being
  /// executed in the work-group in the dimension being vectorized.
  /// @param[in] widestEltTy an optimization hint indicating the widest (vector
  /// element) type in the kernel. Must not be relied on for correctness.
  /// @param[in] factor the vectorization width.
  virtual llvm::Value *createVPKernelWidth(llvm::IRBuilder<> &builder,
                                           llvm::Value *remainingIters,
                                           unsigned widestEltTy,
                                           llvm::ElementCount factor) const {
    (void)builder;
    (void)remainingIters;
    (void)widestEltTy;
    (void)factor;
    return nullptr;
  }

  /// @brief Create a single-source vector shuffle with a general shuffle mask.
  /// Can work with dynamic shuffle masks and scalable vectors, and can return
  /// vectors of a different length to the source.
  ///
  /// @param[in] builder the builder to create the needed instructions
  /// @param[in] src the source vector
  /// @param[in] mask the shuffle mask
  /// @param[in] evl 'Effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes.
  ///
  /// @return the result of the shuffle operation
  virtual llvm::Value *createVectorShuffle(llvm::IRBuilder<> &builder,
                                           llvm::Value *src, llvm::Value *mask,
                                           llvm::Value *evl) const;

  /// @brief Create a vector slide-up operation, that moves all vector elements
  /// up by one place, with the specified element inserted into the zeroth
  /// position.
  ///
  /// @param[in] builder the builder to create the needed instructions
  /// @param[in] src the source vector
  /// @param[in] insert the value to slide into the vacant position
  /// @param[in] evl 'Effective vector length' of the operation. Must be
  /// pre-scaled for vector operations. If evl is null, the operation is not
  /// length-predicated: it executes on all lanes.
  ///
  /// @return the result of the slide-up operation
  virtual llvm::Value *createVectorSlideUp(llvm::IRBuilder<> &builder,
                                           llvm::Value *src,
                                           llvm::Value *insert,
                                           llvm::Value *evl) const;

  /// @brief Determine whether the specified group of interleaved memory
  /// instructions can be optimized or not.
  ///
  /// @param[in] val Memory access operation.
  /// @param[in] kind Kind of interleaved instructions.
  /// @param[in] stride Stride of the interleaved memory operations.
  /// @param[in] groupSize Number of interleaved operations in the group.
  ///
  /// @return true if the interleaved group can be optimized, false otherwise.
  virtual bool canOptimizeInterleavedGroup(const llvm::Instruction &val,
                                           InterleavedOperation kind,
                                           int stride,
                                           unsigned groupSize) const;

  /// @brief Try to optimize a group of consecutive interleaved vector memory
  /// instructions. These instructions collectively access a consecutive chunk
  /// of memory and are sorted by increasing address.
  ///
  /// @note Pointers are scalar and need to be 'scalar aligned'.
  /// @param[in] builder Builder used to create IR.
  /// @param[in] Kind Kind of interleaved group to look for.
  /// @param[in] group List of interleaved operations.
  /// @param[in] masks List of mask operands.
  /// @param[in] baseAddress Base pointer for the memory operation.
  /// @param[in] stride Stride of the interleaved memory operations.
  ///
  /// @return Return true if the interleaved group was optimized or false.
  virtual bool optimizeInterleavedGroup(llvm::IRBuilder<> &builder,
                                        InterleavedOperation Kind,
                                        llvm::ArrayRef<llvm::Value *> group,
                                        llvm::ArrayRef<llvm::Value *> masks,
                                        llvm::Value *baseAddress,
                                        int stride) const;

  /// @brief (De-)interleave a list of vectors.
  ///
  /// @param[in] builder Builder used to generate new instructions.
  /// @param[in,out] vectors List of vectors to (de-)interleave.
  /// @param[in] forward true to interleave, false to deinterleave.
  ///
  /// @return true if the vectors were (de-)interleaved, false otherwise.
  virtual bool interleaveVectors(llvm::IRBuilder<> &builder,
                                 llvm::MutableArrayRef<llvm::Value *> vectors,
                                 bool forward) const;

  /// @brief Estimates the widest SIMD width that will fit into registers for a
  ///        given set of values.
  ///
  /// @param[in] TTI the Target Transform Info
  /// @param[in] vals Set of values to fit into registers
  /// @param[in] width the widest SIMD width to consider
  /// @return the widest SIMD width that is expected to fit into registers, or
  ///         zero if the set can never fit into registers.
  virtual unsigned estimateSimdWidth(
      const llvm::TargetTransformInfo &TTI,
      const llvm::ArrayRef<const llvm::Value *> vals, unsigned width) const;

  /// @brief Get the preferred vector width for the given scalar type
  ///
  /// @param[in] TTI the Target Transform Info
  /// @param[in] Ty the scalar type to get the width for
  /// @return the preferred vector width
  virtual unsigned getVectorWidthForType(const llvm::TargetTransformInfo &TTI,
                                         const llvm::Type &Ty) const;

  /// @brief Return whether the value can be packetized by the given width.
  ///
  /// @param[in] Val The value to be packetized
  /// @param[in] Width The vectorization factor by which to packetize Val
  /// @return true if the value can be packetized, false otherwise.
  virtual bool canPacketize(const llvm::Value *Val,
                            llvm::ElementCount Width) const;

  /// @return Whether a given vector type would be legal as the result of a
  /// binary vp intrinsic.
  virtual bool isVPVectorLegal(const llvm::Function &F, llvm::Type *Ty) const;

 protected:
  /// @brief This type indicates legality of a VP/Masked memory operation in a
  /// target.
  class VPMemOpLegality {
   public:
    constexpr VPMemOpLegality() = default;
    constexpr VPMemOpLegality(bool VPLegal, bool MaskLegal)
        : VPLegal(VPLegal), MaskLegal(MaskLegal) {}

    /// @brief States whether the operation is legal as or not a VP intrinsic.
    void setVPLegality(bool Legal) { VPLegal = Legal; }

    /// @brief States whether the operation is legal ot not as a masked memory
    /// operation.
    void setMaskLegality(bool Legal) { MaskLegal = Legal; }

    /// @brief Tests whether the operation is legal as a VP intrinsic.
    constexpr bool isVPLegal() const { return VPLegal; }

    /// @brief Tests whether the operation is legal as a masked memory
    /// operation.
    constexpr bool isMaskLegal() const { return MaskLegal; }

   private:
    bool VPLegal = false;
    bool MaskLegal = false;
  };

  /// @brief Create an indices vector to be used in createScalableBroadcast()
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] ty Type of the indices vector.
  /// @param[in] factor Vectorization factor.
  /// @param[in] URem Whether to broadcast a fixed-length vector to a scalable
  /// one or a scalable-vector by a fixed amount.
  /// @param[in] N Name of the value to produce.
  static llvm::Value *createBroadcastIndexVector(llvm::IRBuilder<> &builder,
                                                 llvm::Type *ty,
                                                 llvm::ElementCount factor,
                                                 bool URem,
                                                 const llvm::Twine &N = "");

  /// @return A VPMemOpLegality enum stating whether we can create a vp.load or
  /// a masked.load intrinsic.
  ///
  /// @param[in] F The function in which the instruction will be created.
  /// @param[in] Ty Type of the vector to load.
  /// @param[in] Alignment Alignment of the operation.
  virtual VPMemOpLegality isVPLoadLegal(const llvm::Function *F, llvm::Type *Ty,
                                        unsigned Alignment) const;

  /// @return A VPMemOpLegality enum stating whether we can create a vp.store or
  /// a masked.store intrinsic.
  ///
  /// @param[in] F The function in which the instruction will be created.
  /// @param[in] Ty Type of the vector to store.
  /// @param[in] Alignment Alignment of the operation.
  virtual VPMemOpLegality isVPStoreLegal(const llvm::Function *F,
                                         llvm::Type *Ty,
                                         unsigned Alignment) const;

  /// @return A VPMemOpLegality enum stating whether we can create a vp.gather
  /// or a masked.gather intrinsic.
  ///
  /// @param[in] F The function in which the instruction will be created.
  /// @param[in] Ty Type of the vector to gather.
  /// @param[in] Alignment Alignment of the operation.
  virtual VPMemOpLegality isVPGatherLegal(const llvm::Function *F,
                                          llvm::Type *Ty,
                                          unsigned Alignment) const;

  /// @return A VPMemOpLegality enum stating whether we can create a vp.scatter
  /// or a masked.scatter intrinsic.
  ///
  /// @param[in] F The function in which the instruction will be created.
  /// @param[in] Ty Type of the vector to scatter.
  /// @param[in] Alignment Alignment of the operation.
  virtual VPMemOpLegality isVPScatterLegal(const llvm::Function *F,
                                           llvm::Type *Ty,
                                           unsigned Alignment) const;

  /// @brief Function to check whether a given type is valid as the element type
  /// of a scalable vector used in a VP intrinsic.
  ///
  /// @param[in] Ty The type to be checked.
  virtual bool isLegalVPElementType(llvm::Type *Ty) const;

  /// @brief LLVM target machine that will be used for compilation.
  llvm::TargetMachine *TM_;

 private:
  /// @brief Helper function to check legality of memory operations.
  ///
  /// @return Illegal in LLVM < 13 and check leagality in LLVM >= 13.
  VPMemOpLegality checkMemOpLegality(
      const llvm::Function *F,
      llvm::function_ref<bool(const llvm::TargetTransformInfo &, llvm::Type *,
                              unsigned)>
          Checker,
      llvm::Type *Ty, unsigned Alignment) const;

  /// @brief Create a broadcast of a vector.
  ///
  /// @param[in] builder Builder used to create IR.
  /// @param[in] vector Vector to broadcast.
  /// @param[in] VL Vector length.
  /// @param[in] factor Vectorization factor.
  /// @param[in] URem Whether to broadcast a fixed-length vector to a scalable
  /// one or a scalable-vector by a fixed amount
  llvm::Value *createScalableBroadcast(llvm::IRBuilder<> &builder,
                                       llvm::Value *vector, llvm::Value *VL,
                                       llvm::ElementCount factor,
                                       bool URem) const;
};

/// @brief Caches and returns the TargetInfo for a Module.
class TargetInfoAnalysis : public llvm::AnalysisInfoMixin<TargetInfoAnalysis> {
  friend AnalysisInfoMixin<TargetInfoAnalysis>;

 public:
  struct Result {
    Result(std::unique_ptr<TargetInfo> &&I) : Info(std::move(I)) {}
    /// Handle the invalidation of this information.
    ///
    /// When used as a result of TargetInfoAnalysis this method will be called
    /// when the function this was computed for changes. When it returns false,
    /// the information is preserved across those changes.
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    operator TargetInfo *() { return Info.get(); }
    operator const TargetInfo *() const { return Info.get(); }

    std::unique_ptr<TargetInfo> Info;
  };

  using CallbackFn = std::function<Result(const llvm::Module &)>;

  TargetInfoAnalysis();

  TargetInfoAnalysis(llvm::TargetMachine *TM);

  TargetInfoAnalysis(CallbackFn TICallback) : TICallback(TICallback) {}

  /// @brief Retrieve the TargetInfo for the requested module.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    return TICallback(M);
  }

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "TargetInfo analysis"; }

 private:
  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;

  /// @brief Callback function producing a BuiltinInfo on demand.
  CallbackFn TICallback;
};

std::unique_ptr<TargetInfo> createTargetInfoArm(llvm::TargetMachine *tm);

std::unique_ptr<TargetInfo> createTargetInfoAArch64(llvm::TargetMachine *tm);

std::unique_ptr<TargetInfo> createTargetInfoRISCV(llvm::TargetMachine *tm);

/// @brief Create a new vector target info instance.
/// @param[in] tm LLVM target machine that will be used for compilation, can
/// be NULL if no target data is available.
/// @return The new TargetInfo instance.
std::unique_ptr<TargetInfo> createTargetInfoFromTargetMachine(
    llvm::TargetMachine *tm);

/// @}
}  // namespace vecz

#endif  // VECZ_VECZ_TARGET_INFO_H_INCLUDED
