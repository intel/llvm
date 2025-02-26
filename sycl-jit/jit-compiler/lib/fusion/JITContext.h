//==------- JITContext.h - Context holding data for the JIT compiler -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_FUSION_JITCONTEXT_H
#define SYCL_FUSION_JIT_COMPILER_FUSION_JITCONTEXT_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Hashing.h"
#include "Kernel.h"
#include "Options.h"
#include "Parameter.h"

namespace llvm {
class LLVMContext;
} // namespace llvm

namespace jit_compiler {

using CacheKeyT =
    std::tuple<DeviceArchitecture, std::vector<std::string>,
               std::vector<ParameterIdentity>, BarrierFlags,
               std::vector<ParameterInternalization>, std::vector<JITConstant>,
               // This field of the cache is optional because, if all of the
               // ranges are equal, we will perform no remapping, so that fused
               // kernels can be reused with different lists of equal nd-ranges.
               std::optional<std::vector<NDRange>>>;

///
/// Wrapper around a kernel binary.
class KernelBinary {
public:
  explicit KernelBinary(std::string &&Binary, BinaryFormat Format);

  jit_compiler::BinaryAddress address() const;

  size_t size() const;

  BinaryFormat format() const;

private:
  std::string Blob;

  BinaryFormat Format;
};

///
/// Context to persistenly store information across invocations of the JIT
/// compiler and manage lifetimes of binaries.
class JITContext {

public:
  static JITContext &getInstance() {
    static JITContext Instance{};
    return Instance;
  }

  llvm::LLVMContext *getLLVMContext();

  template <typename... Ts> KernelBinary &emplaceKernelBinary(Ts &&...Args) {
    WriteLockT WriteLock{BinariesMutex};
    return Binaries.emplace_back(std::forward<Ts>(Args)...);
  }

  std::optional<SYCLKernelInfo> getCacheEntry(CacheKeyT &Identifier) const;

  void addCacheEntry(CacheKeyT &Identifier, SYCLKernelInfo &Kernel);

private:
  JITContext();
  ~JITContext() = default;
  JITContext(const JITContext &) = delete;
  JITContext(JITContext &&) = delete;
  JITContext &operator=(const JITContext &) = delete;
  JITContext &operator=(const JITContext &&) = delete;

  // FIXME: Change this to std::shared_mutex after switching to C++17.
  using MutexT = std::shared_timed_mutex;

  using ReadLockT = std::shared_lock<MutexT>;

  using WriteLockT = std::unique_lock<MutexT>;

  std::unique_ptr<llvm::LLVMContext> LLVMCtx;

  MutexT BinariesMutex;

  std::vector<KernelBinary> Binaries;

  mutable MutexT CacheMutex;

  std::unordered_map<CacheKeyT, SYCLKernelInfo> Cache;
};
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_FUSION_JITCONTEXT_H
