//===- JITContext.h - Context holding data for the JIT compiler -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "JITBinaryInfo.h"

namespace jit_compiler {

/// Wrapper around a blob created and owned by the JIT compiler.
class JITBinary {
public:
  explicit JITBinary(std::string &&Binary, BinaryFormat Format);

  // Prevent potentially expensive copies.
  JITBinary(const JITBinary &) = delete;
  JITBinary &operator=(const JITBinary &) = delete;

  // Disallow moving as it could hypothetically invalidate the `BinaryAddress`
  // associated with this object if `std::string` implements small string
  // optimization.
  JITBinary(JITBinary &&) = delete;
  JITBinary &operator=(const JITBinary &&) = delete;

  jit_compiler::BinaryAddress address() const;

  size_t size() const;

  BinaryFormat format() const;

private:
  std::string Blob;

  BinaryFormat Format;
};

/// Context to persistenly store information across invocations of the JIT
/// compiler and manage lifetimes of binaries.
class JITContext {

public:
  static JITContext &getInstance() {
    static JITContext Instance{};
    return Instance;
  }

  template <typename... Ts> JITBinary &emplaceBinary(Ts &&...Args) {
    WriteLockT WriteLock{BinariesMutex};
    auto JBUPtr = std::make_unique<JITBinary>(std::forward<Ts>(Args)...);
    JITBinary &JB = *JBUPtr;
    Binaries.emplace(JB.address(), std::move(JBUPtr));
    return JB;
  }

  void destroyBinary(BinaryAddress Addr) {
    WriteLockT WriteLock{BinariesMutex};
    Binaries.erase(Addr);
  }

private:
  JITContext() = default;
  ~JITContext() = default;
  JITContext(const JITContext &) = delete;
  JITContext(JITContext &&) = delete;
  JITContext &operator=(const JITContext &) = delete;
  JITContext &operator=(const JITContext &&) = delete;

  using MutexT = std::shared_mutex;
  using ReadLockT = std::shared_lock<MutexT>;
  using WriteLockT = std::unique_lock<MutexT>;

  MutexT BinariesMutex;

  std::unordered_map<BinaryAddress, std::unique_ptr<JITBinary>> Binaries;
};
} // namespace jit_compiler
