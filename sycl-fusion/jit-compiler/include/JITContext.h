//==------- JITContext.h - Context holding data for the JIT compiler -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_JITCONTEXT_H
#define SYCL_FUSION_JIT_COMPILER_JITCONTEXT_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Hashing.h"
#include "Kernel.h"
#include "Parameter.h"

namespace llvm {
class LLVMContext;
} // namespace llvm

namespace jit_compiler {

using CacheKeyT =
    std::tuple<std::vector<std::string>, ParamIdentList, int,
               std::vector<ParameterInternalization>, std::vector<JITConstant>>;

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

  std::optional<SYCLKernelInfo> getCacheEntry(CacheKeyT &Identifier) const;

  void addCacheEntry(CacheKeyT &Identifier, SYCLKernelInfo &Kernel);

private:
  // FIXME: Change this to std::shared_mutex after switching to C++17.
  using MutexT = std::shared_timed_mutex;

  using ReadLockT = std::shared_lock<MutexT>;

  using WriteLockT = std::unique_lock<MutexT>;

  std::unique_ptr<llvm::LLVMContext> LLVMCtx;

  MutexT BinariesMutex;

  std::vector<SPIRVBinary> Binaries;

  mutable MutexT CacheMutex;

  std::unordered_map<CacheKeyT, SYCLKernelInfo> Cache;
};
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_JITCONTEXT_H
