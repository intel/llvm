//===------------ ESIMDUtils.h - ESIMD utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for processing ESIMD code.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/GenXIntrinsics/GenXMetadata.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace esimd {

constexpr char ESIMD_MARKER_MD[] = "sycl_explicit_simd";
constexpr char GENX_KERNEL_METADATA[] = "genx.kernels";
// sycl/ext/oneapi/experimental/invoke_simd.hpp::__builtin_invoke_simd
// overloads instantiations:
constexpr char INVOKE_SIMD_PREF[] = "_Z33__regcall3____builtin_invoke_simd";

bool isSlmAllocatorConstructor(const Function &F);
bool isSlmAllocatorDestructor(const Function &F);
bool isSlmInit(const Function &F);
bool isSlmAlloc(const Function &F);
bool isSlmFree(const Function &F);
bool isAssertFail(const Function &F);

bool isNbarrierInit(const Function &F);
bool isNbarrierAllocate(const Function &F);

// Tells whether given function is a ESIMD kernel.
bool isESIMDKernel(const Function &F);
// Tells whether given function is a ESIMD function.
bool isESIMD(const Function &F);
// Tells whether given function is a kernel.
bool isKernel(const Function &F);

/// Reports and error with the message \p Msg concatenated with the optional
/// \p OptMsg if \p Condition is false.
inline void assert_and_diag(bool Condition, StringRef Msg,
                            StringRef OptMsg = "") {
  if (!Condition) {
    auto T = Twine(Msg) + OptMsg;
    llvm::report_fatal_error(T, true /* crash diagnostics */);
  }
}

/// Unwraps a presumably simd* type to extract the native vector type encoded
/// in it. Returns nullptr if failed to do so.
Type *getVectorTyOrNull(StructType *STy);

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

// Turn a MDNode into llvm::value or its subclass.
// Return nullptr if the underlying value has type mismatch.
template <typename Ty = llvm::Value> Ty *getValue(llvm::Metadata *M) {
  if (auto VM = dyn_cast<llvm::ValueAsMetadata>(M))
    if (auto V = dyn_cast<Ty>(VM->getValue()))
      return V;
  return nullptr;
}

// Turn given Value into metadata.
inline llvm::Metadata *getMetadata(llvm::Value *V) {
  return llvm::ValueAsMetadata::get(V);
}

// A functor which updates ESIMD kernel's uint64_t metadata in case it is less
// than the given one. Used in callgraph traversal to update nbarriers or SLM
// size metadata. Update is performed by the '()' operator and happens only
// when given function matches one of the kernels - thus, only reachable kernels
// are updated.
struct UpdateUint64MetaDataToMaxValue {
  Module &M;
  // The uint64_t metadata key to update.
  genx::KernelMDOp Key;
  // The new metadata value. Must be greater than the old for update to happen.
  uint64_t NewVal;
  // Pre-selected nodes from GENX_KERNEL_METADATA which can only potentially be
  // updated.
  SmallVector<MDNode *, 4> CandidatesToUpdate;

  UpdateUint64MetaDataToMaxValue(Module &M, genx::KernelMDOp Key,
                                 uint64_t NewVal);

  void operator()(Function *F) const;
};

// Checks if there are any functions that must be inlined early to simplify
// the ESIMD lowering algorithms. If finds such then it may mark them with
// alwaysinline attribute. The function returns true if at least one of
// functions has changed its attribute to alwaysinline.
bool prepareForAlwaysInliner(Module &M);

} // namespace esimd
} // namespace llvm
