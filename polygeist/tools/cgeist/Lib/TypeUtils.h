//===- type utils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_TYPE_UTILS_H
#define MLIR_TOOLS_MLIRCLANG_TYPE_UTILS_H

#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {
class Type;
}

namespace mlirclang {

llvm::Type *anonymize(llvm::Type *T);
bool isRecursiveStruct(llvm::Type *T, llvm::Type *Meta,
                       llvm::SmallPtrSetImpl<llvm::Type *> &seen);

} // namespace mlirclang

#endif
