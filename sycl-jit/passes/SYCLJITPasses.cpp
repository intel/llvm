//===- SYCLJITPasses.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "materializer/SYCLSpecConstMaterializer.h"

using namespace llvm;

llvm::PassPluginLibraryInfo getSYCLJITPassesPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SYCL-JIT pass library", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "sycl-spec-const-materializer") {
                    return false;
                  }
                  FunctionPassManager FPM;
                  FPM.addPass(SYCLSpecConstMaterializer());
                  MPM.addPass(
                      createModuleToFunctionPassAdaptor(std::move(FPM)));
                  return true;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSYCLJITPassesPluginInfo();
}
