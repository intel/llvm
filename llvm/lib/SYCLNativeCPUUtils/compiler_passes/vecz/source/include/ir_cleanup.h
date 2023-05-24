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

#ifndef VECZ_IR_CLEANUP_H_INCLUDED
#define VECZ_IR_CLEANUP_H_INCLUDED

#include <llvm/ADT/SmallPtrSet.h>

namespace llvm {
class Instruction;
}

namespace vecz {
class IRCleanup {
 public:
  /// @brief Mark the instruction as needing deletion. It will only be deleted
  /// if it is unused. This is used to mark instructions with side-effects
  /// (e.g. call, load, store and leaves) that have been replaced and are no
  /// longer needed. Dead Code Elimination will not touch such instructions.
  ///
  /// @param[in] I Instruction to mark as needing deletion.
  void deleteInstructionLater(llvm::Instruction *I);

  /// @brief Get rid of instructions that have been marked for deletion.
  void deleteInstructions();

  /// @brief Immediately delete an instruction, and replace all uses with undef
  ///
  /// @param[in] I Instruction to delete.
  static void deleteInstructionNow(llvm::Instruction *I);

 private:
  /// @brief Instructions that have been marked for deletion.
  llvm::SmallPtrSet<llvm::Instruction *, 16> InstructionsToDelete;
};

}  // namespace vecz

#endif  // VECZ_VECTORIZATION_UNIT_H_INCLUDED
