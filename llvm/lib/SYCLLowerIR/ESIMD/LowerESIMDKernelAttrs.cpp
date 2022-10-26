//===---- LowerESIMDKernelAttrs - lower __esimd_set_kernel_attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Finds and adds  sycl_explicit_simd attributes to wrapper functions that wrap
// ESIMD kernel functions

#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

#define DEBUG_TYPE "LowerESIMDKernelAttrs"

using namespace llvm;

namespace {
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
    void *Ptr = std::calloc(sz, sizeof(llvm::itanium_demangle::Node *));
    Ptrs.push_back(Ptr);
    return Ptr;
  }

  ~SimpleAllocator() { reset(); }
};
} // namespace

namespace llvm {
PreservedAnalyses
SYCLFixupESIMDKernelWrapperMDPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool Modified = false;
  for (Function &F : M) {
    if (llvm::esimd::isESIMD(F)) {
      // TODO: Keep track of traversed functions to avoid repeating traversals
      // over same function.
      sycl::utils::traverseCallgraphUp(
          &F,
          [&](Function *GraphNode) {
            if (!llvm::esimd::isESIMD(*GraphNode) &&
                llvm::esimd::isKernel(*GraphNode)) {

              // Demangle the caller name to check if the function is called
              // from RoundedRangeKernel.
              StringRef MangledName = GraphNode->getName();
              llvm::itanium_demangle::ManglingParser<SimpleAllocator> Parser(
                  MangledName.begin(), MangledName.end());
              itanium_demangle::Node *AST = Parser.parse();
              if (!AST ||
                  AST->getKind() != itanium_demangle::Node::KSpecialName) {
                return;
              }

              itanium_demangle::OutputBuffer NameBuf;
              AST->print(NameBuf);
              StringRef Name(NameBuf.getBuffer(), NameBuf.getCurrentPosition());

              if (!Name.contains("sycl::_V1::detail::RoundedRangeKernel<")) {
                return;
              }

              GraphNode->setMetadata(
                  llvm::esimd::ESIMD_MARKER_MD,
                  llvm::MDNode::get(GraphNode->getContext(), {}));
              Modified = true;
            }
          },
          false);
    }
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
} // namespace llvm
