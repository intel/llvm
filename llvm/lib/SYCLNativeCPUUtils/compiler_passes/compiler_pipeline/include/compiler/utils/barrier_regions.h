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
/// Barrier regions, used by the WorkItemLoopsPass.

#ifndef COMPILER_UTILS_BARRIER_REGIONS_H_INCLUDED
#define COMPILER_UTILS_BARRIER_REGIONS_H_INCLUDED

#include <compiler/utils/attributes.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <multi_llvm/llvm_version.h>

#include "pass_functions.h"

namespace llvm {
class BasicBlock;
class CallInst;
class DbgDeclareInst;
class FenceInst;
class Function;
class Instruction;
class Module;
class StructType;
class Type;
class Value;
}  // namespace llvm

namespace compiler {
namespace utils {

enum { kBarrier_EndID = 0, kBarrier_FirstID, kBarrier_StartNewID };

class Barrier;
class BuiltinInfo;

template <class T, size_t N>
using OrderedSet =
    llvm::SetVector<T, llvm::SmallVector<T, N>, llvm::SmallPtrSet<T, N>>;

/// @brief Struct to store information about an inter-barrier region.
struct BarrierRegion {
  /// @brief the barrier id of this region
  unsigned id = 0;
  /// @brief the barrier call instruction for this region
  llvm::Instruction *barrier_inst = nullptr;
  /// @brief the entry block of this region
  llvm::BasicBlock *entry = nullptr;

  llvm::DenseSet<llvm::Value *> defs;
  /// @brief barrier crossing uses that are defined in this region
  OrderedSet<llvm::Value *, 16> uses_int;
  /// @brief barrier crossing uses that are defined in another region
  OrderedSet<llvm::Value *, 16> uses_ext;
  /// @brief the blocks in this region
  std::vector<llvm::BasicBlock *> blocks;
  /// @brief the exit blocks of this region
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> barrier_blocks;
  /// @brief the barrier ids of the successor regions
  llvm::SmallVector<unsigned, 4> successor_ids;
  /// @brief the work item execution schedule for this region
  BarrierSchedule schedule = BarrierSchedule::Unordered;
};

using BarrierGraph = llvm::SmallVector<BarrierRegion, 8>;

class Barrier {
 public:
  Barrier(llvm::Module &m, llvm::Function &f, bool IsDebug)
      : live_var_mem_ty_(nullptr),
        size_t_bytes(compiler::utils::getSizeTypeBytes(m)),
        module_(m),
        func_(f),
        is_debug_(IsDebug),
        max_live_var_alignment(0) {}

  /// @brief perform the Barrier Region analysis and kernel splitting
  void Run(llvm::ModuleAnalysisManager &mam);

  /// @brief return whether the barrier struct needs to contain anything
  bool hasLiveVars() const { return !whole_live_variables_set_.empty(); }

  /// @brief returns the StructType of the barrier struct
  llvm::StructType *getLiveVarsType() const { return live_var_mem_ty_; }

  /// @brief returns the maximum alignment of the barrier struct
  unsigned getLiveVarMaxAlignment() const { return max_live_var_alignment; }

  /// @brief gets the split subkernel for the given barrier id
  llvm::Function *getSubkernel(unsigned id) const {
    return kernel_id_map_.find(id)->second;
  }

  /// @brief gets the number of regions/subkernels
  size_t getNumSubkernels() const { return kernel_id_map_.size(); }

  llvm::CallInst *getBarrierCall(unsigned id) const {
    return llvm::dyn_cast_or_null<llvm::CallInst>(
        barrier_graph[id - kBarrier_FirstID].barrier_inst);
  }

  /// @brief gets the size of the fixed sized part of the barrier struct
  size_t getLiveVarMemSizeFixed() const { return live_var_mem_size_fixed; }

  /// @brief gets the minimum size of the scalable part of the barrier struct
  size_t getLiveVarMemSizeScalable() const {
    return live_var_mem_size_scalable;
  }

  /// @brief gets the element index of the first scalable member of the barrier
  /// struct
  size_t getLiveVarMemScalablesIndex() const {
    return live_var_mem_scalables_index;
  }

  /// @brief gets the barrier IDs of the successors of the given barrier region
  const llvm::SmallVectorImpl<unsigned> &getSuccessorIds(unsigned id) const {
    return barrier_graph[id - kBarrier_FirstID].successor_ids;
  }

  /// @brief gets the barrier IDs of the successors of the given barrier region
  BarrierSchedule getSchedule(unsigned id) const {
    return barrier_graph[id - kBarrier_FirstID].schedule;
  }

  /// @brief replaces a subkernel with a given function
  void replaceSubkernel(llvm::Function *from, llvm::Function *to);

  /// @brief Type containing list of debug intrinsics and the source variable
  /// byte offset in the live variables struct.
  // TODO CA-1115 llvm.dbg.declare is being deprecated
  using debug_intrinsics_t =
      llvm::SmallVector<std::pair<llvm::DbgDeclareInst *, unsigned>, 4>;
  const debug_intrinsics_t &getDebugIntrinsics() const {
    return debug_intrinsics_;
  }

#if LLVM_VERSION_GREATER_EQUAL(19, 0)
  using debug_variable_records_t =
      llvm::SmallVector<std::pair<llvm::DbgVariableRecord *, unsigned>, 4>;
  const debug_variable_records_t &getDebugDbgVariableRecords() const {
    return debug_variable_records_;
  }
#endif

  /// @brief gets the original function
  llvm::Function &getFunc() { return func_; }
  const llvm::Function &getFunc() const { return func_; }

  /// @brief struct to help retrieval of values from the barrier struct
  struct LiveValuesHelper {
    const Barrier &barrier;
    /// @brief A cache of queried live-values addresses (inside the live
    /// variables struct), stored by the pair (value, member_idx).
    llvm::DenseMap<std::pair<const llvm::Value *, unsigned>, llvm::Value *>
        live_GEPs;
    llvm::DenseMap<const llvm::Value *, llvm::Value *> reloads;
    llvm::IRBuilder<> gepBuilder;
    llvm::Value *barrier_struct = nullptr;
    llvm::Value *vscale = nullptr;

    LiveValuesHelper(const Barrier &b, llvm::Instruction *i, llvm::Value *s)
        : barrier(b), gepBuilder(i), barrier_struct(s) {}

    LiveValuesHelper(const Barrier &b, llvm::BasicBlock *bb, llvm::Value *s)
        : barrier(b), gepBuilder(bb), barrier_struct(s) {}

    /// @brief Return a GEP instruction pointing to the given value/idx pair in
    /// the barrier struct.
    ///
    /// @return The GEP corresponding to the address of the value in the
    /// struct, or nullptr if the value could not be found in the struct.
    llvm::Value *getGEP(const llvm::Value *live, unsigned member_idx = 0);

    /// @brief Return a GEP instruction corresponding to the address of
    /// the given ExtractValueInst in the barriers struct.
    ///
    /// @return The GEP corresponding to the address of the value in the
    /// struct, or nullptr if the value is not an ExtractValueInst.
    llvm::Value *getExtractValueGEP(const llvm::Value *live);

    /// @brief get a value reloaded from the barrier struct.
    ///
    /// @param[in] live the live value to retrieve from the barrier
    /// @param[in] ir where to insert new instructions
    /// @param[in] name a postfix to append to new value names
    /// @param[in] reuse whether to generate the load for a given value only
    /// once, returning the previously cached value on further requests.
    llvm::Value *getReload(llvm::Value *live, llvm::IRBuilderBase &ir,
                           const char *name, bool reuse = false);
  };

 private:
  /// @brief The first is set for livein and the second is set for liveout
  using live_in_out_t =
      std::pair<llvm::DenseSet<llvm::Value *>, llvm::DenseSet<llvm::Value *>>;
  /// @brief Type for memory allocation of live variables at all of barriers
  using live_variable_mem_t = OrderedSet<llvm::Value *, 32>;
  /// @brief Type for index of live variables on live variable information
  /// Indexed by the pair (value, member_idx)
  using live_variable_index_map_t =
      llvm::DenseMap<std::pair<const llvm::Value *, unsigned>, unsigned>;
  /// @brief Type for index of live variables on live variable information
  /// Indexed by the pair (value, member_idx)
  using live_variable_scalables_map_t = live_variable_index_map_t;
  /// @brief Type for ids of barriers
  using barrier_id_map_t = llvm::DenseMap<llvm::BasicBlock *, unsigned>;
  /// @brief Type for ids of new kernel functions
  using kernel_id_map_t = llvm::DenseMap<unsigned, llvm::Function *>;
  /// @brief Type for map from ids to fence instructions
  using fence_id_map_t = llvm::DenseMap<unsigned, llvm::FenceInst *>;
  /// @brief Type between block and instruction for barrier.
  using barrier_block_inst_map_t =
      llvm::DenseMap<llvm::BasicBlock *, llvm::Instruction *>;
  /// @brief Type between block and block for barrier.
  using barrier_block_block_set_t = llvm::DenseSet<llvm::BasicBlock *>;
  /// @brief Type between barrier id and stub call instructions. First
  /// component of the pair is invoked before the barrier, the second after.
  using debug_stub_map_t =
      llvm::DenseMap<unsigned, std::pair<llvm::CallInst *, llvm::CallInst *>>;

  /// @brief Keep whole live variables at all of barriers.
  live_variable_mem_t whole_live_variables_set_;
  /// @brief Keep index of live variables on live variable information.
  live_variable_index_map_t live_variable_index_map_;
  /// @brief Keep offsets of scalable live variables.
  live_variable_scalables_map_t live_variable_scalables_map_;
  /// @brief Keep ids of barriers.
  barrier_id_map_t barrier_id_map_;
  /// @brief Keep ids of barriers.
  kernel_id_map_t kernel_id_map_;
  /// @brief Keep struct types for live variables' memory layout.
  llvm::StructType *live_var_mem_ty_;
  /// @brief The total size of the non-scalable barrier struct
  size_t live_var_mem_size_fixed = 0;
  /// @brief The total unscaled size of the scalable barrier struct
  size_t live_var_mem_size_scalable = 0;
  /// @brief The index of the scalables buffer array in the barrier struct.
  size_t live_var_mem_scalables_index = 0;
  /// @brief Keep barriers.
  llvm::SmallVector<llvm::CallInst *, 8> barriers_;
  /// @brief Set of basic blocks that have a barrier as their successor
  barrier_block_block_set_t barrier_successor_set_;
  /// @brief Map between barrier ids and call instructions invoking stubs
  debug_stub_map_t barrier_stub_call_map_;
  /// @brief List of debug intrinsics and byte offsets into live variable struct
  debug_intrinsics_t debug_intrinsics_;
#if LLVM_VERSION_GREATER_EQUAL(19, 0)
  /// @brief List of debug DbgVariableRecords and byte offsets into live
  /// variable struct
  debug_variable_records_t debug_variable_records_;
#endif

  size_t size_t_bytes;

  BarrierGraph barrier_graph;

  llvm::Module &module_;
  llvm::Function &func_;

  BuiltinInfo *bi_ = nullptr;

  /// @brief Set to true if we want to debug the kernel. This involves adding
  /// debug stub functions and an extra alloca to aide debugging.
  const bool is_debug_;

  // @brief max alignment required for the live variables.
  unsigned max_live_var_alignment;

  /// @brief Find Barriers.
  void FindBarriers();

  /// @brief Split block with barrier.
  void SplitBlockwithBarrier();

  /// @brief Generate an empty kernel that only duplicates the source kernel's
  /// CFG
  ///
  /// This is used to do a "dry run" of kernel splitting in order to obtain the
  /// dominator tree, which is needed for correct identification of values that
  /// cross the barrier.
  ///
  /// @param[in] region the region to clone into the new kernel.
  /// @param[out] bbmap a mapping of original blocks onto the empty clones.
  /// @return the fake kernel
  llvm::Function *GenerateFakeKernel(
      BarrierRegion &region,
      llvm::DenseMap<llvm::BasicBlock *, llvm::BasicBlock *> &bbmap);

  /// @brief Obtain a set of Basic Blocks for an inter-barrier region
  ///
  /// It traverses the CFG, following successors, until it hits a barrier,
  /// building the region's internal data.
  ///
  /// @param[out] region the region to process
  void GatherBarrierRegionBlocks(BarrierRegion &region);

  /// @brief Obtain a set of Values used in a region that cross a barrier
  ///
  /// A value use crosses a barrier in the following cases:
  /// * Its use is not in the same region as the defintion
  /// * Its definition does not dominate the use
  ///
  /// @param[in] region The inter-barrier region
  /// @param[in] ignore set of values to ignore
  void GatherBarrierRegionUses(BarrierRegion &region,
                               llvm::DenseSet<llvm::Value *> &ignore);

  /// @brief Find livein and liveout variables per each basic block.
  void FindLiveVariables();

  /// @brief Remove variables that are better recalculated than stored in the
  ///        barrier, for instance casts and vector splats.
  void TidyLiveVariables();

  /// @brief Pad the field types to an alignment by adding an int array if
  /// needed
  /// @param field_tys The vector of types representing the final structure
  /// @param offset The current offset in the structure
  /// @param alignment The required alignment
  /// @return The new offset (or original offset if no padding needed)
  unsigned PadTypeToAlignment(llvm::SmallVectorImpl<llvm::Type *> &field_tys,
                              unsigned offset, unsigned alignment);

  /// @brief Make type for whole live variables.
  void MakeLiveVariableMemType();

  /// @brief Generate new kernel from an inter-barrier region such that no call
  /// to barriers occur within it.
  ///
  /// @param[in] region the inter-barrier region to create the kernel from
  /// @return the new kernel
  llvm::Function *GenerateNewKernel(BarrierRegion &region);

  /// @brief This function is a copy from llvm::CloneBasicBlock. In order to
  /// update live variable information, some of codes are added.
  ///
  /// @param[in] bb Basic block to copy.
  /// @param[out] vmap Map for value for cloning.
  /// @param[in] name_suffix Name for suffix.
  /// @param[out] live_defs_info Live definitions' info current basic block.
  /// @param[in] F Current function.
  ///
  /// @return Return cloned basic block.
  llvm::BasicBlock *CloneBasicBlock(llvm::BasicBlock *bb,
                                    llvm::ValueToValueMapTy &vmap,
                                    const llvm::Twine &name_suffix,
                                    live_variable_mem_t &live_defs_info,
                                    llvm::Function *F);

  /// @brief Seperate kernel function with barrier boundary.
  void SeperateKernelWithBarrier();
};

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_BARRIER_REGIONS_H_INCLUDED
