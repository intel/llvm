//=== PolygeistBase.td - Base definitions for polygeist dialect -*- tablegen =//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_BASE
#define POLYGEIST_BASE

include "mlir/IR/OpBase.td"

def Polygeist_Dialect : Dialect {
  let name = "polygeist";
  let cppNamespace = "::mlir::polygeist";
  let description = [{}];
}

#endif // POLYGEIST_BASE
