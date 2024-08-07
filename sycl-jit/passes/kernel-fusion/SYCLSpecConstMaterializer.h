//==--------------------- SYCLSpecConstMaterializer.h ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_SPEC_CONST_MATERIALIZER_H
#define SYCL_SPEC_CONST_MATERIALIZER_H

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include <map>
#include <set>

namespace llvm {
class Function;

///
/// Utility pass to insert specialization constants values into the module as a
/// metadata node.
class SYCLSpecConstDataInserter
    : public PassInfoMixin<SYCLSpecConstDataInserter> {
public:
  SYCLSpecConstDataInserter(ArrayRef<unsigned char> SpecConstData)
      : SpecConstData(SpecConstData) {};

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  ArrayRef<unsigned char> SpecConstData;
};

///
/// Pass to materialize specialization constants. Specialization constants
/// represent constants whose values can be set dynamically during execution of
/// the SYCL application. The values of these constants are fixed when a SYCL
/// kernel function is invoked, and they do not change during the execution of
/// the kernel. This pass receives the values of all specialization constants
/// used by a kernel and materializes them as concrete types. This is done in
/// order to be able to enable other optimization opportunities (SCCP, SROA and
/// CSE), we do not track instructions that can be removed as a result of
/// materialization, as the pipeline runs DCE pass afterwords.
class SYCLSpecConstMaterializer
    : public PassInfoMixin<SYCLSpecConstMaterializer> {
public:
  SYCLSpecConstMaterializer() : SpecConstData(nullptr), SpecConstDataSize(0) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);

private:
  // Main entry point, checks for implicit specialization constant kernel
  // argument and, if present, performs the optimizations.
  PreservedAnalyses handleKernel(Function &Kernel);

  bool readMetadata();

  // Collects all the uses of the specialization constant kernel argument.
  // This results with TypesAndOffsets and LoadsToType being populated.
  void populateUses(Argument *A);

  // Use TypesAndOffsets to allocate global variables of given types which get
  // initialized with value taken from the specialization constant blob at a
  // given offset.
  void allocateSpecConstant(StringRef KernelName);

  // Re-write uses of loads from the specialization constant kernel argument to
  // the global variable.
  void fixupSpecConstantUses();

  // Walk down all uses of a given GEP instruction in order to find loads from
  // the offsetted pointer.
  SmallVector<LoadInst *> collectGEPsLoads(GetElementPtrInst *GEP);

  // Helper to report debug message (if enabled) and reset the state.
  void reportAndReset();

private:
  // Run time known values of specialization constants passed from SYCL rt,
  // data pointer and size.
  const unsigned char *SpecConstData;
  size_t SpecConstDataSize;

  // Module the current function belongs to.
  Module *Mod{nullptr};

  // Type of the specialization constant and the offset into the SpecConstBlob,
  // at which the value is located.
  using TypeAtOffset = std::pair<Type *, uint64_t>;

  // Unique uses of spec const (type and offset).
  std::set<TypeAtOffset> TypesAndOffsets{};
  // A map from type and offset to a specialization constant blob to a
  // GlobalVariable containing its value.
  std::map<TypeAtOffset, GlobalVariable *> TypesAndOffsetsToBlob{};
  // A map of load instruction to its type and offset to a specialization
  // constant blob.
  std::map<LoadInst *, TypeAtOffset> LoadsToTypes{};
};
} // namespace llvm

#endif // SYCL_SPEC_CONST_MATERIALIZER_H
