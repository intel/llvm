//=== PolygeistTypes.td - Polygeist dialect type definitions *- tablegen -*===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for Polygeist dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_TYPES
#define POLYGEIST_TYPES

include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
include "mlir/Dialect/Polygeist/IR/PolygeistBase.td"
include "mlir/IR/BuiltinTypes.td"

class Polygeist_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Polygeist_Dialect, name,
              !listconcat([MemRefElementTypeInterface,
                           LLVM_PointerElementTypeInterface],
                               traits)> {
  let mnemonic = typeMnemonic;
}

def Polygeist_StructType :
    Polygeist_Type<"Struct", "struct"> {
  let summary = "Polygeist Struct";
  let description = [{
    Type used to represent a collection of data members together in memory.
  }];
  let parameters = (ins OptionalArrayRefParameter<"mlir::Type">:$body,
                        OptionalParameter<"std::optional<StringAttr>">:$name,
                        DefaultValuedParameter<"bool", "false">:$isPacked,
                        DefaultValuedParameter<"bool", "false">:$isOpaque);
  let builders = [
    TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$body),[{
      return $_get($_ctxt, body, std::nullopt, false, false);
    }]>,
    TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$body, "std::optional<StringAttr>":$name, "bool":$isPacked),[{
      return $_get($_ctxt, body, name, isPacked, false);
    }]>
  ];
  let extraClassDeclaration = [{
    bool isPacked() { return getIsPacked(); }
    bool isOpaque() { return getIsOpaque(); }
    static StructType getOpaque(StringAttr name, mlir::MLIRContext *context) {
      return get(context, std::nullopt, name, false, true);
    }
  }];
  let assemblyFormat = [{
    `<` ($name^ `,` ` `)?
        (`isPacked` `=` $isPacked^ ` `)?
        (`isOpaque` `=` $isOpaque^)?
        (`(` $body^ `)`)? `>`
  }];
  let genVerifyDecl = 1;
}

#endif // POLYGEIST_TYPES
