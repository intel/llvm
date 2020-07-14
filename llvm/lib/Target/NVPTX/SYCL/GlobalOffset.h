//===---------- GlobalOffset.h - Global Offset Support for CUDA ---------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It looks for uses
// of the `llvm.nvvm.implicit.offset` intrinsic and replaces it with a offset
// parameter which will be threaded through from the kernel entry point.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_GLOBALOFFSET_H
#define LLVM_SYCL_GLOBALOFFSET_H

#include "llvm/Pass.h"

namespace llvm {

ModulePass *createGlobalOffsetPass();

} // end namespace llvm

#endif
