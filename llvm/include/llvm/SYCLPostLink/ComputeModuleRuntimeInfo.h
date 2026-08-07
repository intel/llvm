//===- ComputeModuleRuntimeInfo.h - compute runtime info for module -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functions for computing module properties and symbols for SYCL modules.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/PropertySetIO.h"
#include <string>
namespace llvm {

class Function;

namespace sycl {

struct GlobalBinImageProps {
  bool EmitKernelParamInfo;
  bool EmitProgramMetadata;
  bool EmitKernelNames;
  bool EmitExportedSymbols;
  bool EmitImportedSymbols;
  bool EmitDeviceGlobalPropSet;
};

inline bool isModuleUsingAsan(const Module &M) {
  return any_of(M.globals(), [](const GlobalVariable &GV) {
    return GV.getName().starts_with("__AsanKernelMetadata");
  });
}

inline bool isModuleUsingMsan(const Module &M) {
  return any_of(M.globals(), [](const GlobalVariable &GV) {
    return GV.getName().starts_with("__MsanKernelMetadata");
  });
}

inline bool isModuleUsingTsan(const Module &M) {
  return any_of(M.globals(), [](const GlobalVariable &GV) {
    return GV.getName().starts_with("__TsanKernelMetadata");
  });
}
using PropSetRegTy = llvm::util::PropertySetRegistry;
using EntryPointSet = SetVector<Function *>;

PropSetRegTy computeDeviceLibProperties(const Module &M,
                                        const std::string &SYCLDeviceLibName);

PropSetRegTy computeModuleProperties(const Module &M,
                                     const EntryPointSet &EntryPoints,
                                     const GlobalBinImageProps &GlobProps,
                                     bool AllowDeviceImageDependencies,
                                     int IdQueriesRange);

std::string computeModuleSymbolTable(const Module &M,
                                     const EntryPointSet &EntryPoints);

} // namespace sycl
} // namespace llvm
