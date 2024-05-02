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

#ifndef COMPILER_UTILS_ATTRIBUTES_H_INCLUDED
#define COMPILER_UTILS_ATTRIBUTES_H_INCLUDED

#include <llvm/ADT/StringRef.h>

#include <optional>

namespace llvm {
class CallInst;
class Function;
}  // namespace llvm

namespace compiler {
namespace utils {

/// @brief Encodes information that a function is a kernel
///
/// @param[in] F Function in which to encode the information.
void setIsKernel(llvm::Function &F);

/// @brief Encodes information that a function is a kernel entry point
///
/// @param[in] F Function in which to encode the information.
void setIsKernelEntryPt(llvm::Function &F);

/// @brief Returns whether the function is a kernel under compilation.
///
/// @param[in] F Function to check.
bool isKernel(const llvm::Function &F);

/// @brief Returns whether the function is a kernel entry point under
/// compilation.
///
/// @param[in] F Function to check.
bool isKernelEntryPt(const llvm::Function &F);

/// @brief Drops any information about whether a function is a kernel.
///
/// @param[in] F Function to drop information from.
void dropIsKernel(llvm::Function &F);

/// @brief Takes information about kernels from one function to another.
///
/// Removes information from the old function, and overwrites any such
/// information in the new function.
///
/// @param[in] ToF Function to copy to.
/// @param[in] FromF Function to copy from.
void takeIsKernel(llvm::Function &ToF, llvm::Function &FromF);

/// @brief Sets the original function name as an attribute.
void setOrigFnName(llvm::Function &F);

/// @brief Retrieves the original function name from the given Function.
///
/// @return The original function name (via function attributes) or an empty
/// string if none is found.
llvm::StringRef getOrigFnName(const llvm::Function &F);

/// @brief Retrieves the original function name from the given Function, or the
/// Function's name.
///
/// @return The original function name (via function attributes) or the
/// function's name if none is found.
llvm::StringRef getOrigFnNameOrFnName(const llvm::Function &F);

/// @brief Sets the original function name as an attribute.
void setBaseFnName(llvm::Function &F, llvm::StringRef N);

/// @brief Retrieves the base function name component from the given Function.
///
/// @return The base function name (via function attributes) or an empty string
/// if none is found.
llvm::StringRef getBaseFnName(const llvm::Function &F);

/// @brief Retrieves the base function name component from the given Function,
/// or the Function's name.
///
/// @return The base function name (via function attributes) or the function's
/// name if none is found.
llvm::StringRef getBaseFnNameOrFnName(const llvm::Function &F);

/// @brief Retrieves the base function name from the given Function and
/// sets it if none is found.
/// @param F The function to read "base function name" attributes from
/// @param SetFromF The function whose name is set as F's base function
/// name if none is found in F.
llvm::StringRef getOrSetBaseFnName(llvm::Function &F,
                                   const llvm::Function &SetFromF);

/// @brief Sets the local memory usage estimation for the given function.
///
/// @param[in] F the function in which to add the attribute
/// @param[in] LocalMemUsage the (estimated) local memory usage in bytes
void setLocalMemoryUsage(llvm::Function &F, uint64_t LocalMemUsage);

/// @brief Gets the local memory usage estimation for the given function.
///
/// @param[in] F Function from which to pull the attribute
/// @return the (estimated) local memory usage in bytes if present,
/// std::nullopt otherwise.
std::optional<uint64_t> getLocalMemoryUsage(const llvm::Function &F);

/// @brief Sets information about a function's required DMA size as an
/// attribute.
///
/// @param[in] F Function in which to add the attribute.
/// @param[in] DMASizeBytes DMA size in bytes.
void setDMAReqdSizeBytes(llvm::Function &F, uint32_t DMASizeBytes);

/// @brief Retrieves information about a function's required DMA size as an
/// attribute.
///
/// @param[in] F Function from which to pull the attribute
/// @return The required DMA size order if present, else `std::nullopt`
std::optional<uint32_t> getDMAReqdSizeBytes(const llvm::Function &F);

/// @brief Determines the ordering of work item execution after a barrier.
enum class BarrierSchedule {
  /// @brief The barrier pass is free to schedule work items in any order.
  Unordered = 0,
  /// @brief The barrier region is entirely uniform (no dependence on work item
  /// ID) such that execution of multiple work items is redundant and we are
  /// free to execute the region for only a single work item. Additionally,
  /// such a region is not allowed to read from or write to the barrier struct
  /// (the region cannot use any variables defined outwith it, nor define any
  /// variables used outwith it). Used by work group collectives to initialize
  /// their accumulators.
  Once,
  /// @brief The barrier region should execute all vectorized work items first,
  /// followed by the scalar tail.
  ScalarTail,
  /// @brief The barrier region must be executed in Local Linear ID order.
  Linear,
};

/// @brief Sets the work item execution schedule for the given barrier.
///
/// @param[in] CI the barrier call instruction
/// @param[in] Sched the execution schedule to set
void setBarrierSchedule(llvm::CallInst &CI, BarrierSchedule Sched);

/// @brief Gets the work item execution schedule for the given barrier.
///
/// @param[in] CI the barrier call instruction
/// @return the execution schedule for this barrier
BarrierSchedule getBarrierSchedule(const llvm::CallInst &CI);

/// @brief Marks a kernel's subgroups as degenerate
///
/// @param[in] F Function in which to encode the information.
void setHasDegenerateSubgroups(llvm::Function &F);

/// @brief Returns whether the kernel has degenerate subgroups.
///
/// @param[in] F Function to check.
bool hasDegenerateSubgroups(const llvm::Function &F);

/// @brief Marks a function as not explicitly using subgroups
///
/// May be set even with unresolved external functions, assuming those don't
/// explicitly use subgroups.
///
/// @param[in] F Function in which to encode the information.
void setHasNoExplicitSubgroups(llvm::Function &F);

/// @brief Returns whether the kernel does not explicitly use subgroups
///
/// @param[in] F Function to check.
bool hasNoExplicitSubgroups(const llvm::Function &F);

/// @brief Returns the mux subgroup size for the current function.
///
/// Currently always returns 1!
unsigned getMuxSubgroupSize(const llvm::Function &F);

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_ATTRIBUTES_H_INCLUDED
