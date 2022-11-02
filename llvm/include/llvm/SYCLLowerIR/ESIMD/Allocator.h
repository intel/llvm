//===--------- Allocator.h - Allocator for demangler
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Demangler allocator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_ALLOCATOR_H
#define LLVM_SYCLLOWERIR_ALLOCATOR_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/ItaniumDemangle.h"

namespace llvm {

// Simplest possible implementation of an allocator for the Itanium demangler
class SimpleAllocator {
protected:
  SmallVector<void *, 128> Ptrs;

public:
  void reset() {
    for (void *Ptr : Ptrs) {
      // Destructors are not called, but that is OK for the
      // itanium_demangle::Node subclasses
      std::free(Ptr);
    }
    Ptrs.resize(0);
  }

  template <typename T, typename... Args> T *makeNode(Args &&...args) {
    void *Ptr = std::calloc(1, sizeof(T));
    Ptrs.push_back(Ptr);
    return new (Ptr) T(std::forward<Args>(args)...);
  }

  void *allocateNodeArray(size_t sz) {
    void *Ptr = std::calloc(sz, sizeof(itanium_demangle::Node *));
    Ptrs.push_back(Ptr);
    return Ptr;
  }

  ~SimpleAllocator() { reset(); }
};

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_ALLOCATOR_H
