//==------------------------ SYCLFusionPasses.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "Kernel.h"

#include "kernel-fusion/SYCLSpecConstMaterializer.h"

using namespace llvm;
using namespace jit_compiler;

llvm::PassPluginLibraryInfo getSYCLKernelFusionPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SYCL-Module-Info", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sycl-spec-const-materializer") {
                    FunctionPassManager FPM;
                    FPM.addPass(SYCLSpecConstMaterializer());
                    MPM.addPass(
                        createModuleToFunctionPassAdaptor(std::move(FPM)));
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSYCLKernelFusionPluginInfo();
}
