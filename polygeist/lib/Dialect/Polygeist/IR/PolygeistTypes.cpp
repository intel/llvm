//===--- PolygeistTypes.cpp -----------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/IR/PolygeistTypes.h"

using namespace mlir;
using namespace mlir::polygeist;

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 llvm::ArrayRef<Type> body,
                                 std::optional<StringAttr> name, bool isPacked,
                                 bool isOpaque) {
  if (isOpaque && !name)
    return emitError() << "opaque struct type must have a name";
  return success();
}
