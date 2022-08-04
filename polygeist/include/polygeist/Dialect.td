//===- BFVDialect.td - BFV dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_DIALECT
#define POLYGEIST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BFV dialect definition.
//===----------------------------------------------------------------------===//

def Polygeist_Dialect : Dialect {
  let name = "polygeist";
  let description = [{}];
  let cppNamespace = "::mlir::polygeist";
}

//===----------------------------------------------------------------------===//
// Base BFV operation definition.
//===----------------------------------------------------------------------===//

class Polygeist_Op<string mnemonic, list<Trait> traits = []>
    : Op<Polygeist_Dialect, mnemonic, traits>;

#endif // POLYGEIST_DIALECT
