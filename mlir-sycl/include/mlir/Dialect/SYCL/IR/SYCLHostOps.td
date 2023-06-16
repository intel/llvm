//===-- SYCLHostOps.td - SYCL dialect operation definitions *- tablegen -*-===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for SYCL dialect operations involving
// host code.
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_HOST_OPS
#define SYCL_HOST_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

////////////////////////////////////////////////////////////////////////////////
// BASE DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

class SYCL_HostOp<string mnemonic, list<Trait> traits = []> :
    SYCL_Op<"host." # mnemonic, traits>;

////////////////////////////////////////////////////////////////////////////////
// OPERATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCLHostConstructorOp : SYCL_HostOp<"constructor"> {
  let summary = [{
    Operation representing the construction of a SYCL object, i.e., allocation,
    member initialization and possible casts to the required address space.
  }];
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs LLVM_AnyPointer:$res);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // SYCL_HOST_OPS
