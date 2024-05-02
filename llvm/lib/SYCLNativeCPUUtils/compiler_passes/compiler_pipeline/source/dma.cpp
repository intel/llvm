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

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/dma.h>
#include <compiler/utils/pass_functions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <multi_llvm/multi_llvm.h>

#include <array>

namespace compiler {
namespace utils {

llvm::Value *isThreadEQ(llvm::BasicBlock *bb, unsigned x, unsigned y,
                        unsigned z, llvm::Function &LocalIDFn) {
  llvm::IRBuilder<> builder(bb);
  LocalIDFn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
  auto *const indexType = LocalIDFn.arg_begin()->getType();
  llvm::Value *result = llvm::ConstantInt::getTrue(bb->getContext());

  const std::array<unsigned, 3> threadIDs{x, y, z};
  for (unsigned i = 0; i < threadIDs.size(); ++i) {
    auto *const index = llvm::ConstantInt::get(indexType, i);
    auto *const localID = builder.CreateCall(&LocalIDFn, index);
    localID->setCallingConv(LocalIDFn.getCallingConv());

    auto *thread =
        llvm::ConstantInt::get(LocalIDFn.getReturnType(), threadIDs[i]);
    auto *const cmp = builder.CreateICmpEQ(localID, thread);
    result = (i == 0) ? cmp : builder.CreateAnd(result, cmp);
  }

  return result;
}

llvm::Value *isThreadZero(llvm::BasicBlock *BB, llvm::Function &LocalIDFn) {
  return isThreadEQ(BB, 0, 0, 0, LocalIDFn);
}

void buildThreadCheck(llvm::BasicBlock *entryBlock, llvm::BasicBlock *trueBlock,
                      llvm::BasicBlock *falseBlock, llvm::Function &LocalIDFn) {
  // only thread 0 in the work group should execute the DMA.
  llvm::IRBuilder<> entryBuilder(entryBlock);
  entryBuilder.CreateCondBr(isThreadZero(entryBlock, LocalIDFn), trueBlock,
                            falseBlock);
}

llvm::StructType *getOrCreateMuxDMAEventType(llvm::Module &m) {
  if (auto *eventType = llvm::StructType::getTypeByName(
          m.getContext(), MuxBuiltins::dma_event_type)) {
    return eventType;
  }

  return llvm::StructType::create(m.getContext(), MuxBuiltins::dma_event_type);
}
}  // namespace utils
}  // namespace compiler
