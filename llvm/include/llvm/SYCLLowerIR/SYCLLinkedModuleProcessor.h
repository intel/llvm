//===-- SYCLLinkedModuleProcessor.h - finalize a fully linked module ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file contains a number of functions to create a pass that can be called
// by the LTO backend that will finalize a fully-linked module.
//===----------------------------------------------------------------------===//
#pragma once
#include "SpecConstants.h"
namespace llvm {

class PassRegistry;
class ModulePass;
ModulePass *
    createSYCLLinkedModuleProcessorPass(llvm::SpecConstantsPass::HandlingMode);
void initializeSYCLLinkedModuleProcessorPass(PassRegistry &);

} // namespace llvm
