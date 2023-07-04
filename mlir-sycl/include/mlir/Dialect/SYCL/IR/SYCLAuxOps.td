//===-- SYCLAuxOps.td - Auxiliary ops for the SYCL dialect -*- tablegen -*-===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for operations in the SYCL dialect that do not
// model language constructs, but are required for interoperability with the
// upstream dialects.
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_AUX_OPS
#define SYCL_AUX_OPS

////////////////////////////////////////////////////////////////////////////////
// BASE DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

// Single-element API classes that can be represented by a built-in type.
def SingleElementClass : AnyTypeOf<[SYCL_HalfType, SYCL_VecType]>;

// Base class for auxiliary operations.
class SYCL_AuxOp<string mnemonic, list<Trait> traits = []> :
    SYCL_Op<"mlir." # mnemonic, traits>;

////////////////////////////////////////////////////////////////////////////////
// OPERATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCLWrapOp : SYCL_AuxOp<"wrap",
    [DeclareOpInterfaceMethods<CastOpInterface>, Pure]> {
  let summary = "Wraps a value into a SYCL-specific type.";
  let description = [{
    This operation wraps a value in one of MLIR's built-in types into a SYCL
    type with a single, matching body type.

    Example:
    ```
    %1 = sycl.mlir.wrap %0 : f16 to !sycl.half<(f16)>
    ```
  }];
  
  let arguments = (ins AnyType:$source);
  let results = (outs SingleElementClass:$result);
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];
  let hasFolder = 1;
}

def SYCLUnwrapOp : SYCL_AuxOp<"unwrap",
    [DeclareOpInterfaceMethods<CastOpInterface>, Pure]> {
  let summary = "Unwraps a value from a SYCL-specific type.";
  let description = [{
    This operation unwraps a value in one of MLIR's built-in types from a SYCL
    type with a single, matching body type.

    Example:
    ```
    %1 = sycl.mlir.unwrap %0 : !sycl.half<(f16)> to f16
    ```
  }];
  
  let arguments = (ins SingleElementClass:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];
  let hasFolder = 1;
}

#endif // SYCL_AUX_OPS
