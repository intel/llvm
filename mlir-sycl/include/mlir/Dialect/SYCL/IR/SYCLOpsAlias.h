// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsAlias.h ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_ALIAS_H_
#define MLIR_SYCL_OPS_ALIAS_H_

#include "mlir/IR/OpImplementation.h"

class SYCLOpAsmInterface final : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

public:
  AliasResult getAlias(mlir::Attribute Attr, llvm::raw_ostream &OS) const final;
  AliasResult getAlias(mlir::Type Type, llvm::raw_ostream &OS) const final;
};

#endif
