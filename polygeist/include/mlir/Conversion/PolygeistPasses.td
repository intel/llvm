//=== PolygeistPasses.td - Polygeist dialect passes definitions -*- tablegen =//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_POLYGEISTPASSES
#define MLIR_CONVERSION_POLYGEISTPASSES

include "mlir/Pass/PassBase.td"

def ConvertPolygeistToLLVM : Pass<"convert-polygeist-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Polygeist dialect to LLVM dialect";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect"];
  let options = [
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
            /*default=*/"false", "Generate LLVM IR using opaque pointers "
            "instead of typed pointers">,
  ];
}

#endif // MLIR_CONVERSION_POLYGEISTPASSES
