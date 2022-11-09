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

namespace clang {
class QualType;
class RecordType;

namespace CodeGen {
class CodeGenModule;
}
} // namespace clang

namespace mlirclang {
namespace CodeGen {
class CodeGenTypes;
}
} // namespace mlirclang

namespace mlir {
class IntegerAttr;
class MLIRContext;
class Type;
} // namespace mlir

namespace llvm {
class Type;
}

namespace mlirclang {

llvm::Type *anonymize(llvm::Type *T);

bool isRecursiveStruct(llvm::Type *T, llvm::Type *Meta,
                       llvm::SmallPtrSetImpl<llvm::Type *> &Seen);

mlir::IntegerAttr wrapIntegerMemorySpace(unsigned MemorySpace,
                                         mlir::MLIRContext *Ctx);

mlir::Type getSYCLType(const clang::RecordType *RT,
                       mlirclang::CodeGen::CodeGenTypes &CGT);

llvm::Type *getLLVMType(clang::QualType QT, clang::CodeGen::CodeGenModule &CGM);

} // namespace mlirclang

#endif // MLIR_TOOLS_MLIRCLANG_TYPE_UTILS_H
