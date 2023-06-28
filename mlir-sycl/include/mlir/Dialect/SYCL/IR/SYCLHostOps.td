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
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

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
    and member initialization.

    This operation differs with `sycl.constructor` in two points:
    - This operation will return a pointer to the constructed object instead of
      receiving a reference to the object to construct
    - This operation will return an `llvm.ptr` to any type instead of requiring a
      `sycl` type.

    These two differences are the reason why this operation was introduced in
    the first place: this is a short-term solution to represent construction of
    a SYCL object in the host side. Using a `sycl.constructor` operation would
    imply performing heavy modifications to the host LLVM code (or blurring the
    semantics of the `sycl.constructor` operation).

    Note that, despite the more relaxed typing, the `type` attribute still needs
    to be a type in the `sycl` dialect.
  }];
  let arguments = (ins Variadic<AnyType>:$args, Builtin_TypeAttr:$type);
  let results = (outs LLVM_AnyPointer:$res);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = true;
}

def SYCLHostKernelNameOp
    : SYCL_HostOp<"kernel_name",
        [Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{
    Operation representing a constant holding the name of a kernel, i.e., a
    `gpu.func` with the `kernel` attribute.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, SymbolRefAttr:$kernel_name);
  let assemblyFormat = [{
    $sym_name `->` $kernel_name attr-dict
  }];
}

def SYCLHostGetKernelOp
    : SYCL_HostOp<"get_kernel",
        [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{
    Operation that defines a reference to a kernel, i.e., a `gpu.func` with the
    `kernel` attribute.
  }];
  let arguments = (ins SymbolRefAttr:$kernel_name);
  let results = (outs LLVM_AnyPointer:$res);
  let assemblyFormat = [{
    $kernel_name attr-dict `:` type($res)
  }];
}

#endif // SYCL_HOST_OPS
