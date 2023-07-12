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

def SYCLHostConstructorOp : SYCL_HostOp<"constructor",
    [MemoryEffectsOpInterface]> {
  let summary = [{
    Operation representing the construction of a SYCL object, i.e., allocation,
    and member initialization.

    This operation differs from `sycl.constructor` as it will take a
    `llvm.ptr` to any type instead of requiring a memref of a `sycl` type.

    This difference is the reason why this operation was introduced in
    the first place: this is a short-term solution to represent construction of
    a SYCL object in the host side. Using a `sycl.constructor` operation would
    imply performing heavy modifications to the host LLVM code (or blurring the
    semantics of the `sycl.constructor` operation).

    Note that, despite the more relaxed typing, the `type` attribute still needs
    to be a type in the `sycl` dialect.
  }];
  let arguments = (ins LLVM_AnyPointer:$dst, Variadic<AnyType>:$args,
                        Builtin_TypeAttr:$type);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
      /// Return the memory effects associated to each argument.
      ///
      /// The first argument, `this` will have `MemoryEffects::Write` memory
      /// effect associated and the rest of argument will have
      /// `MemoryEffects::Read` if they're a pointer/memref and no memory effect
      /// otherwise.
      void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                          ::mlir::MemoryEffects::Effect>> &effects);
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

def SYCLHostHandlerSetKernel
    : SYCL_HostOp<"handler.set_kernel",
        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{
    Operation assigning a kernel to a `sycl::handler`, thus pairing the handler
    and the kernel being launched.
  }];
  let arguments = (ins LLVM_AnyPointer:$handler, SymbolRefAttr:$kernel_name);
  let assemblyFormat = [{
    $handler `->` $kernel_name attr-dict `:` type($handler)
  }];
}

def SYCLHostHandlerSetNDRange : SYCL_HostOp<"handler.set_nd_range"> {
  let summary = [{
    Operation assigning an nd-range to a `sycl::handler`, setting the nd-range
    of the kernel to be launched.

    The input nd-range can be specified either with a memref to a
    `sycl.nd_range` or a memref to a `sycl.range` (global size) and an optional
    memref to a `sycl.id` (offset).
  }];
  let arguments = (ins LLVM_AnyPointer:$handler, Variadic<AnyType>:$nd_range);
  let assemblyFormat = [{
    $handler `->` $nd_range attr-dict `:` type(operands)
  }];
  let hasVerifier = 1;
}

def SYCLHostSetCaptured : SYCL_HostOp<"set_captured"> {
  let summary = "Marks a value as captured by a kernel function object.";
  let description = [{
    This operation represents that the given `value` was captured in the kernel
    function object `lambda` at index `index`. The op is created during the 
    progressive raising towards the `sycl.host.launch_kernel` op.

    Example:
    ```
    sycl.host.set_captured $lambda[2] = %accessor : !llvm.ptr, !llvm.ptr
    ```
  }];
  let arguments = (ins LLVM_AnyPointer:$lambda, I64Attr:$index, AnyType:$value);
  let assemblyFormat = [{
    $lambda `[` $index `]` `=` $value attr-dict `:` type(operands)
  }];
}

def SYCLHostLaunchKernel : SYCL_HostOp<"launch_kernel",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     AttrSizedOperandSegments]> {
  let summary = "Launches a SYCL kernel.";
  let description = [{
    This operation represents the launch of a kernel function with the given
    range and kernel arguments. Its purpose is to collect the information
    surfaced by the host-raising process about a particular launch, thereby
    providing an entry-point for host-device analyses and optimizations.

    If the `range` operand is not present, this operation represents a
    *single_task* launch, otherwise a *parallel_for* invocation. The `range`
    operand either accepts an `nd_range` memref or a `range` memref. In the
    latter case, an `id` memref can be optionally given as the `offset`.

    Note that due to the current focus on host-device optimizations for
    individual launches, this operation currently does not model the queue or
    the command-group handler associated with the launch, nor does it observe or
    yield any events.

    Example:
    ```
    sycl.host.launch_kernel @kernels::@k0[%1](%2, %3)
    : (memref<?x!sycl_range_1_>, memref<?x!sycl_accessor_1_i32_r_gb>, i32) -> ()
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel_name,
    Optional<MemRefOf<[SYCL_NdRangeType, SYCL_RangeType]>>:$range,
    Optional<MemRefOf<[SYCL_IDType]>>:$offset, Variadic<AnyType>:$args);
  let assemblyFormat = [{
    $kernel_name (`[` $range^ (`,` $offset^)? `]`)? `` (`(` $args^ `)`)?
      attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // SYCL_HOST_OPS
