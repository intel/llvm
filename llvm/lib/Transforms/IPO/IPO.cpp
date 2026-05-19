//===-- IPO.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the common infrastructure (including C bindings) for
// libLLVMIPO.a, which implements several transformations over the LLVM
// intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"

using namespace llvm;

void llvm::initializeIPO(PassRegistry &Registry) {
  initializeAlwaysInlinerLegacyPassPass(Registry);
  initializeBarrierNoopPass(Registry);
  initializeDAEPass(Registry);
  initializeDAHPass(Registry);
<<<<<<< HEAD
  initializeDAESYCLPass(Registry);
=======
  initializeExpandVariadicsPass(Registry);
>>>>>>> a5077468984ac3c47e6a3ca779c6f0ba680706c0
  initializeGlobalDCELegacyPassPass(Registry);
  initializeLoopExtractorLegacyPassPass(Registry);
  initializeSingleLoopExtractorPass(Registry);
}
