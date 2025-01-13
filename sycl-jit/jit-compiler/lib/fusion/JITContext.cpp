//==---------------------------- JITContext.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITContext.h"
#include "llvm/IR/LLVMContext.h"

using namespace jit_compiler;

KernelBinary::KernelBinary(std::string &&Binary, BinaryFormat Fmt)
    : Blob{std::move(Binary)}, Format{Fmt} {}

jit_compiler::BinaryAddress KernelBinary::address() const {
  // FIXME: Verify it's a good idea to perform this reinterpret_cast here.
  return reinterpret_cast<jit_compiler::BinaryAddress>(Blob.c_str());
}

size_t KernelBinary::size() const { return Blob.size(); }

BinaryFormat KernelBinary::format() const { return Format; }

JITContext::JITContext() : LLVMCtx{new llvm::LLVMContext}, Binaries{} {}

llvm::LLVMContext *JITContext::getLLVMContext() { return LLVMCtx.get(); }

std::optional<SYCLKernelInfo>
JITContext::getCacheEntry(CacheKeyT &Identifier) const {
  ReadLockT ReadLock{CacheMutex};
  auto Entry = Cache.find(Identifier);
  if (Entry != Cache.end()) {
    return Entry->second;
  }
  return {};
}

void JITContext::addCacheEntry(CacheKeyT &Identifier, SYCLKernelInfo &Kernel) {
  WriteLockT WriteLock{CacheMutex};
  Cache.emplace(Identifier, Kernel);
}
