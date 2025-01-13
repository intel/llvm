//===-- SYCLConditionalCallOnDevice.h - SYCLConditionalCallOnDevice Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass performs transformations on functions which represent the conditional
// call to application's callable object. The conditional call is based on the
// SYCL device's aspects or architecture passed to the functions.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_CONDITIONAL_CALL_ON_DEVICE_H
#define LLVM_SYCL_CONDITIONAL_CALL_ON_DEVICE_H

#include "llvm/IR/PassManager.h"

#include <string>

namespace llvm {

class SYCLConditionalCallOnDevicePass
    : public PassInfoMixin<SYCLConditionalCallOnDevicePass> {
public:
  SYCLConditionalCallOnDevicePass(std::string SYCLUniquePrefix = "")
      : UniquePrefix(SYCLUniquePrefix) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  std::string UniquePrefix;
};

} // namespace llvm

#endif // LLVM_SYCL_CONDITIONAL_CALL_ON_DEVICE_H
