//==------ PostLinkActions.h - Fork of sycl-post-link actions for RTC ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_JIT_COMPILER_RTC_POST_LINK_ACTIONS_H
#define SYCL_JIT_COMPILER_RTC_POST_LINK_ACTIONS_H

#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>

namespace jit_compiler::post_link {

using namespace llvm;

template <class PassClass> bool runModulePass(Module &M) {
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MPM.addPass(PassClass{});
  PreservedAnalyses Res = MPM.run(M, MAM);
  return !Res.areAllPreserved();
}

// Removes the global variable "llvm.used" and returns true on success.
// "llvm.used" is a global constant array containing references to kernels
// available in the module and callable from host code. The elements of
// the array are ConstantExpr bitcast to i8*.
// The variable must be removed as it is a) has done the job to the moment
// of this function call and b) the references to the kernels callable from
// host must not have users.
bool removeSYCLKernelsConstRefArray(Module &M);

// Removes all device_global variables from the llvm.compiler.used global
// variable. A device_global with internal linkage will be in llvm.compiler.used
// to avoid the compiler wrongfully removing it during optimizations. However,
// as an effect the device_global variables will also be distributed across
// binaries, even if llvm.compiler.used has served its purpose. To avoid
// polluting other binaries with unused device_global variables, we remove them
// from llvm.compiler.used and erase them if they have no further uses.
bool removeDeviceGlobalFromCompilerUsed(llvm::Module &M);

} // namespace jit_compiler::post_link

#endif // SYCL_JIT_COMPILER_RTC_POST_LINK_ACTIONS_H
