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
using namespace llvm::esimd;

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
