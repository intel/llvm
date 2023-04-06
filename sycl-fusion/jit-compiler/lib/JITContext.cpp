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

SPIRVBinary::SPIRVBinary(std::string Binary) : Blob{std::move(Binary)} {}

jit_compiler::BinaryAddress SPIRVBinary::address() const {
  // FIXME: Verify it's a good idea to perform this reinterpret_cast here.
  return reinterpret_cast<jit_compiler::BinaryAddress>(Blob.c_str());
}

size_t SPIRVBinary::size() const { return Blob.size(); }

JITContext::JITContext() : LLVMCtx{new llvm::LLVMContext}, Binaries{} {}

JITContext::~JITContext() = default;

llvm::LLVMContext *JITContext::getLLVMContext() { return LLVMCtx.get(); }

SPIRVBinary &JITContext::emplaceSPIRVBinary(std::string Binary) {
  WriteLockT WriteLock{BinariesMutex};
  // NOTE: With C++17, which returns a reference from emplace_back, the
  // following code would be even simpler.
  Binaries.emplace_back(std::move(Binary));
  return Binaries.back();
}

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
