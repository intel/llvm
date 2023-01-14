//==------- JITContext.h - Context holding data for the JIT compiler -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_JITCONTEXT_H
#define SYCL_FUSION_JIT_COMPILER_JITCONTEXT_H

#include "llvm/IR/LLVMContext.h"
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "Kernel.h"
#include "Parameter.h"

namespace jit_compiler {

///
/// Wrapper around a SPIR-V binary.
class SPIRVBinary {
public:
  explicit SPIRVBinary(std::string Binary);

  jit_compiler::BinaryAddress address() const;

  size_t size() const;

private:
  std::string Blob;
};

///
/// Context to persistenly store information across invocations of the JIT
/// compiler and manage lifetimes of binaries.
class JITContext {

public:
  JITContext();

  ~JITContext();

  llvm::LLVMContext *getLLVMContext();

  SPIRVBinary &emplaceSPIRVBinary(std::string Binary);

private:
  // FIXME: Change this to std::shared_mutex after switching to C++17.
  using MutexT = std::shared_timed_mutex;

  using ReadLockT = std::shared_lock<MutexT>;

  using WriteLockT = std::unique_lock<MutexT>;

  std::unique_ptr<llvm::LLVMContext> LLVMCtx;

  MutexT BinariesMutex;

  std::vector<SPIRVBinary> Binaries;
};
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_JITCONTEXT_H
