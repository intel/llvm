//===- LocalAccessorToSharedMemory.cpp - Local Accessor Support for CUDA --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It modifies
// kernel entry points which take pointers to shared memory and modifies them
// to take offsets into shared memory (represented by a symbol in the shared
// address space). The SYCL runtime is expected to provide offsets rather than
// pointers to these functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_LOCALACCESSORTOSHAREDMEMORY_H
#define LLVM_SYCL_LOCALACCESSORTOSHAREDMEMORY_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createLocalAccessorToSharedMemoryPass();

} // end namespace llvm

#endif
