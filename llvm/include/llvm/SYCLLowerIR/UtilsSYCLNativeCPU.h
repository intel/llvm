//===----- UtilsSYCLNativeCPU.h - Pass pipeline for SYCL Native CPU ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions and constants for SYCL Native CPU.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"

#include "llvm/IR/Constants.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PropertySetIO.h"

namespace llvm {
namespace sycl {
namespace utils {

// Used to schedule passes in the device compiler cc1 invocation for
// Native CPU.
void addSYCLNativeCPUEarlyPasses(ModulePassManager &MPM);

void addSYCLNativeCPUBackendPasses(ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM,
                                   OptimizationLevel OptLevel);

constexpr char SYCLNATIVECPUSUFFIX[] = ".SYCLNCPU";
constexpr char SYCLNATIVECPUKERNEL[] = ".NativeCPUKernel";
constexpr char SYCLNATIVECPUPREFIX[] = "__dpcpp_nativecpu";
inline llvm::Twine addSYCLNativeCPUSuffix(StringRef S) {
  if (S.starts_with(SYCLNATIVECPUPREFIX) || S.ends_with(SYCLNATIVECPUKERNEL))
    return S;
  return llvm::Twine(S, SYCLNATIVECPUSUFFIX);
}

constexpr unsigned SyclNativeCpuLocalAS = 3;

/// Adds Native CPU declarations to the module so that they can be
/// referenced in the binary.
inline Function *addDeclarationForNativeCPU(StringRef Name, LLVMContext &C,
                                            Module &M) {
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(C),
      {PointerType::getUnqual(C), PointerType::getUnqual(C)}, false);
  auto FCalle = M.getOrInsertFunction(
      sycl::utils::addSYCLNativeCPUSuffix(Name).str(), FTy);
  Function *F = dyn_cast<Function>(FCalle.getCallee());
  if (F == nullptr)
    report_fatal_error("Unexpected callee");
  return F;
}

/// Adds declarations of NativeCPU kernels to the binary in a format so that
/// the NativeCPU UR adapter can extract the pointers and invoke the kernels.
/// Shared by OffloadWrapper and ClangLinkerWrapper.
template <class Wrapper, class T>
auto addDeclarationsForNativeCPU(
    Wrapper &Wrapp, const std::optional<util::PropertySet> &NativeCPUProps,
    const llvm::MemoryBuffer *MB, llvm::LLVMContext &C, llvm::Module &M,
    const T &addPropertySetToModule, StructType *SyclPropSetTy)
    -> decltype(addPropertySetToModule(*NativeCPUProps)) {

  // the Native CPU UR adapter expects the BinaryStart field to point to
  //
  // struct nativecpu_program {
  //   nativecpu_entry *entries;
  //   ur_program_properties_t *properties;
  // };
  //
  // where "entries" is an array of:
  //
  // struct nativecpu_entry {
  //   char *kernelname;
  //   unsigned char *kernel_ptr;
  // };
  auto getPtrTy = [&C]() { return PointerType::getUnqual(C); };

  StructType *NCPUProgramT =
      StructType::create({getPtrTy(), getPtrTy()}, "nativecpu_program");
  StructType *NCPUEntryT =
      StructType::create({getPtrTy(), getPtrTy()}, "__nativecpu_entry");
  SmallVector<Constant *, 5> NativeCPUEntries;
  for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI) {
    auto *NewDecl = sycl::utils::addDeclarationForNativeCPU(*LI, C, M);
    NativeCPUEntries.push_back(ConstantStruct::get(
        NCPUEntryT,
        {Wrapp.addStringToModule(*LI, "__ncpu_function_name"), NewDecl}));
  }

  // Add an empty entry that we use as end iterator
  auto *NativeCPUEndStr =
      Wrapp.addStringToModule("__nativecpu_end", "__ncpu_end_str");
  auto *NullPtr = llvm::ConstantPointerNull::get(getPtrTy());
  NativeCPUEntries.push_back(
      ConstantStruct::get(NCPUEntryT, {NativeCPUEndStr, NullPtr}));

  // Create the constant array containing the {kernel name, function pointers}
  // pairs
  ArrayType *ATy = ArrayType::get(NCPUEntryT, NativeCPUEntries.size());
  Constant *CA = ConstantArray::get(ATy, NativeCPUEntries);
  auto *GVar = new GlobalVariable(M, CA->getType(), true,
                                  GlobalVariable::InternalLinkage, CA,
                                  "__sycl_native_cpu_decls");
  auto *EntriesBegin = ConstantExpr::getGetElementPtr(
      GVar->getValueType(), GVar, Wrapp.getSizetConstPair(0u, 0u));

  // Add Native CPU specific properties to the nativecpu_program struct
  Constant *PropValue = NullPtr;
  if (NativeCPUProps.has_value()) {
    auto PropsOrErr = addPropertySetToModule(*NativeCPUProps);
    std::pair<Constant *, Constant *> Props;
    if constexpr (std::is_same_v<decltype(PropsOrErr),
                                 std::pair<Constant *, Constant *>>) {
      Props = PropsOrErr;
    } else {
      if (!PropsOrErr)
        return PropsOrErr.takeError();
      Props = PropsOrErr.get();
    }
    auto *Category = Wrapp.addStringToModule(
        /*sycl::PropSetRegTy*/
        llvm::util::PropertySetRegistry::SYCL_NATIVE_CPU_PROPS,
        "SYCL_PropSetName");
    auto S =
        ConstantStruct::get(SyclPropSetTy, Category, Props.first, Props.second);
    auto T = Wrapp.addStructArrayToModule({S}, SyclPropSetTy);
    PropValue = T.first;
  }

  // Create the nativecpu_program struct.
  // We add it to a ConstantArray of length 1 because the SYCL runtime expects
  // a non-zero sized binary image, and this allows it to point the end of the
  // binary image to the end of the array.
  auto *Program = ConstantStruct::get(NCPUProgramT, {EntriesBegin, PropValue});
  ArrayType *ProgramATy = ArrayType::get(NCPUProgramT, 1);
  Constant *CPA = ConstantArray::get(ProgramATy, {Program});
  auto *ProgramGVar =
      new GlobalVariable(M, ProgramATy, true, GlobalVariable::InternalLinkage,
                         CPA, "__sycl_native_cpu_program");
  auto *ProgramBegin =
      ConstantExpr::getGetElementPtr(ProgramGVar->getValueType(), ProgramGVar,
                                     Wrapp.getSizetConstPair(0u, 0u));
  auto *ProgramEnd =
      ConstantExpr::getGetElementPtr(ProgramGVar->getValueType(), ProgramGVar,
                                     Wrapp.getSizetConstPair(0u, 1u));
  return std::make_pair(ProgramBegin, ProgramEnd);
}

} // namespace utils
} // namespace sycl
} // namespace llvm
