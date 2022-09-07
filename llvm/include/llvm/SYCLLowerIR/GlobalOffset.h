//===---------- GlobalOffset.h - Global Offset Support for CUDA ---------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_GLOBALOFFSET_H
#define LLVM_SYCL_GLOBALOFFSET_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"

namespace llvm {

class ModulePass;
class PassRegistry;

/// This pass operates on SYCL kernels that target AMDGPU or NVVM. It looks for
/// uses of the `llvm.{amdgcn|nvvm}.implicit.offset` intrinsic and replaces it
/// with an offset parameter which will be threaded through from the kernel
/// entry point.
class GlobalOffsetPass : public PassInfoMixin<GlobalOffsetPass> {
private:
  using KernelPayload = TargetHelpers::KernelPayload;
  using ArchType = TargetHelpers::ArchType;

public:
  explicit GlobalOffsetPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static StringRef getPassName() { return "Add implicit SYCL global offset"; }

private:
  /// After the execution of this function, the module to which the kernel
  /// `Func` belongs, contains both the original function and its clone with the
  /// signature extended with the implicit offset parameter and `_with_offset`
  /// appended to the name.
  /// An alloca of 3 zeros (corresponding to offsets in x, y and z) is added to
  /// the original kernel, in order to keep the interface of kernel's call
  /// graph unified, regardless of the fact if the global offset has been used.
  ///
  /// \param Func Kernel to be processed.
  void processKernelEntryPoint(Function *Func);

  /// This function adds an implicit parameter to the function containing a
  /// call instruction to the implicit offset intrinsic or another function
  /// (which eventually calls the instrinsic). If the call instruction is to
  /// the implicit offset intrinsic, then the intrinisic is replaced with the
  /// parameter that was added.
  ///
  /// Once the function, say `F`, containing a call to `Callee` has the
  /// implicit parameter added, callers of `F` are processed by recursively
  /// calling this function, passing `F` to `CalleeWithImplicitParam`.
  ///
  /// Since the cloning of entry points may alter the users of a function, the
  /// cloning must be done as early as possible, as to ensure that no users are
  /// added to previous callees in the call-tree.
  ///
  /// \param Callee is the function (to which this transformation has already
  /// been applied), or to the implicit offset intrinsic.
  ///
  /// \param CalleeWithImplicitParam indicates whether Callee is to the
  /// implicit intrinsic (when `nullptr`) or to another function (not
  /// `nullptr`) - this is used to know whether calls to it needs to have the
  /// implicit parameter added to it or replaced with the implicit parameter.
  void addImplicitParameterToCallers(Module &M, Value *Callee,
                                     Function *CalleeWithImplicitParam);

  /// For a given function `Func` extend signature to contain an implicit
  /// offset argument.
  ///
  /// \param Func A function to add offset to.
  ///
  /// \param ImplicitArgumentType Architecture dependant type of the implicit
  /// argument holding the global offset.
  ///
  /// \param KeepOriginal If set to true, rather than splicing the old `Func`,
  /// keep it intact and create a clone of it with `_wit_offset` appended to
  /// the name.
  ///
  /// \returns A pair of new function with the offset argument added and a
  /// pointer to the implicit argument (either a func argument or a bitcast
  /// turning it to the correct type).
  std::pair<Function *, Value *>
  addOffsetArgumentToFunction(Module &M, Function *Func,
                              Type *ImplicitArgumentType = nullptr,
                              bool KeepOriginal = false);

  /// Create a mapping of kernel entry points to their metadata nodes. While
  /// iterating over kernels make sure that a given kernel entry point has no
  /// llvm uses.
  ///
  /// \param KernelPayloads A collection of kernel functions present in a
  /// module `M`.
  ///
  /// \returns A map of kernel functions to corresponding metadata nodes.
  DenseMap<Function *, MDNode *>
  generateKernelMDNodeMap(Module &M,
                          SmallVectorImpl<KernelPayload> &KernelPayloads);

private:
  /// Keep track of which functions have been processed to avoid processing
  /// twice.
  llvm::DenseMap<Function *, Value *> ProcessedFunctions;
  /// Keep a map of all entry point functions with metadata.
  llvm::DenseMap<Function *, MDNode *> EntryPointMetadata;
  /// A type of implicit argument added to the kernel signature.
  llvm::Type *KernelImplicitArgumentType = nullptr;
  /// A type used for the alloca holding the values of global offsets.
  llvm::Type *ImplicitOffsetPtrType = nullptr;

  ArchType AT;
  unsigned TargetAS = 0;
};

ModulePass *createGlobalOffsetPassLegacy();
void initializeGlobalOffsetLegacyPass(PassRegistry &);

} // end namespace llvm

#endif
