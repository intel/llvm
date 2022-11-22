//===- type utils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_TYPE_UTILS_H
#define MLIR_TOOLS_MLIRCLANG_TYPE_UTILS_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/BuiltinTypes.h"
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

unsigned getAddressSpace(mlir::Type Ty);

/// Given a MemRefType or LLVMPointerType, change the element type, keeping the
/// rest of the parameters.
mlir::Type getPtrTyWithNewType(mlir::Type Orig, mlir::Type NewElementType);

mlir::Type getSYCLType(const clang::RecordType *RT,
                       mlirclang::CodeGen::CodeGenTypes &CGT);

llvm::Type *getLLVMType(clang::QualType QT, clang::CodeGen::CodeGenModule &CGM);

bool isFPOrFPVectorTy(mlir::Type Ty);
bool isIntOrIntVectorTy(mlir::Type Ty);

inline bool isPointerOrMemRefTy(mlir::Type Ty) {
  return Ty.isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>();
}

inline bool isFirstClassType(mlir::Type Ty) {
  return Ty.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                mlir::VectorType, mlir::MemRefType, mlir::LLVM::LLVMPointerType,
                mlir::LLVM::LLVMStructType>() ||
         mlir::sycl::isSYCLType(Ty);
}

inline bool isAggregateType(mlir::Type Ty) {
  return Ty.isa<mlir::LLVM::LLVMStructType>() || mlir::sycl::isSYCLType(Ty);
}

unsigned getPrimitiveSizeInBits(mlir::Type Ty);

} // namespace mlirclang

#endif // MLIR_TOOLS_MLIRCLANG_TYPE_UTILS_H
