//===- SPIRVLLVMUtil.h - SPIR-V LLVM-specific Utility Functions -*- C++ -*-===//
//
// Has inclusions from the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines utility functions dedicated to processing LLVM classes
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVLLVMUTIL_H
#define SPIRV_LIBSPIRV_SPIRVLLVMUTIL_H

#include "llvm/IR/Constants.h"
#include "llvm/Support/Casting.h"

namespace SPIRV {
inline bool isManifestConstant(const llvm::Constant *C) {
  if (llvm::isa<llvm::ConstantData>(C)) {
    return true;
  } else if (llvm::isa<llvm::ConstantAggregate>(C) ||
             llvm::isa<llvm::ConstantExpr>(C)) {
    for (const llvm::Value *Subc : C->operand_values()) {
      if (!isManifestConstant(llvm::cast<llvm::Constant>(Subc)))
        return false;
    }
    return true;
  }
  return false;
}
} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVLLVMUTIL_H
