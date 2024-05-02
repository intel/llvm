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
/// LLVM DMA pass utility functions.

#ifndef COMPILER_UTILS_DMA_H_INCLUDED
#define COMPILER_UTILS_DMA_H_INCLUDED

#include <llvm/ADT/Twine.h>
#include <llvm/IR/IRBuilder.h>

#include <functional>

namespace llvm {
class BasicBlock;
class Module;
class Value;
}  // namespace llvm

namespace compiler {
namespace utils {

class BIMuxInfoConcept;

/// @addtogroup utils
/// @{

/// @brief Helper function to check the local ID of the current thread.
///
/// @param[in] bb Basic block to generate the check in.
/// @param[in] x The local id in the x dimension to compare against.
/// @param[in] y The local id in the y dimension to compare against.
/// @param[in] z The local id in the z dimension to compare against.
/// @param[in] GetLocalIDFn Function used to get the local work-item ID
///
/// @return A true Value if the local ID equals that passed via the index
/// arguments, false otherwise.
llvm::Value *isThreadEQ(llvm::BasicBlock *bb, unsigned x, unsigned y,
                        unsigned z, llvm::Function &GetLocalIDFn);

/// @brief Helper function to check if the local ID of the current thread is {0,
/// 0, 0}.
///
/// @param[in] bb Basic block to generate the check in.
/// @param[in] GetLocalIDFn Function used to get the local work-item ID
///
/// @return A true Value if the local ID is {0, 0, 0} / false otherwise.
llvm::Value *isThreadZero(llvm::BasicBlock *bb, llvm::Function &GetLocalIDFn);

/// @brief Insert 'thread-checking' logic in the entry block, so that control
/// branches to the 'true' block when the current work-item in the first in the
/// work-group (e.g. ID zero in all dimensions) or to the 'false' block for
/// other work-items
///
/// @param[in] entryBlock Block to insert the 'thread-checking' logic
/// @param[in] trueBlock Block to execute only on the first work-item
/// @param[in] falseBlock Block to execute on all other work-items
/// @param[in] GetLocalIDFn Function used to get the local work-item ID
void buildThreadCheck(llvm::BasicBlock *entryBlock, llvm::BasicBlock *trueBlock,
                      llvm::BasicBlock *falseBlock,
                      llvm::Function &GetLocalIDFn);

/// @brief Gets or creates the __mux_dma_event_t type.
///
/// This type may be declared by other passes hence we "get or create it".
///
/// @param[in] m LLVM Module to get or create the type in.
///
/// @return The opaque struct declaration of the __mux_dma_event_t type.
llvm::StructType *getOrCreateMuxDMAEventType(llvm::Module &m);

/// @}
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_DMA_H_INCLUDED
