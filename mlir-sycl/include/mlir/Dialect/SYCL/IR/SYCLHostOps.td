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
// TRAIT DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCLHostHandlerOp : SYCLOpTrait<"SYCLHostHandlerOp">;

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
        [SYCLHostHandlerOp, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{
    Operation assigning a kernel to a `sycl::handler`, thus pairing the handler
    and the kernel being launched.
  }];
  let arguments = (ins LLVM_AnyPointer:$handler, SymbolRefAttr:$kernel_name);
  let assemblyFormat = [{
    $handler `->` $kernel_name attr-dict `:` type($handler)
  }];
}

def SYCLHostHandlerSetNDRange
    : SYCL_HostOp<"handler.set_nd_range", [SYCLHostHandlerOp]> {
  let summary = [{
    Operation assigning an nd-range to a `sycl::handler`, setting the nd-range
    of the kernel to be launched.

    The `range` operand expects an `nd_range` pointer or a `range` pointer. In
    the latter case, an `id` pointer can be optionally given as the `offset`. In
    the former case, the `nd_range` attribute must be set.
  }];
  let arguments = (ins LLVM_AnyPointer:$handler,
                       LLVM_AnyPointer:$range,
                       Optional<LLVM_AnyPointer>:$offset,
                       UnitAttr:$nd_range);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$handler,
                   "::mlir::Value":$range,
                   "bool":$nd_range), [{
      auto ndRangeAttr = nd_range ? $_builder.getUnitAttr() : UnitAttr();
      build($_builder, $_state, handler, range, /*offset=*/Value(), ndRangeAttr);
    }]>,
    OpBuilder<(ins "::mlir::Value":$handler,
                   "::mlir::Value":$range,
                   "::mlir::Value":$offset), [{
      build($_builder, $_state, handler, range, offset);
    }]>
  ];
  let assemblyFormat = [{
    $handler `->` (`nd_range` $nd_range^):(`range`)? $range
    (`,` `offset` $offset^)? attr-dict `:` type(operands)
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
    sycl.host.set_captured %lambda[2] = %accessor : !llvm.ptr, !llvm.ptr
    ```
  }];
  let arguments = (ins LLVM_AnyPointer:$lambda, I64Attr:$index, AnyType:$value);
  let assemblyFormat = [{
    $lambda `[` $index `]` `=` $value attr-dict `:` type(operands)
  }];
}

def SYCLHostScheduleKernel : SYCL_HostOp<"schedule_kernel",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     AttrSizedOperandSegments]> {
  let summary = "Schedules a SYCL kernel launch.";
  let description = [{
    This operation represents the scheduling of a launch of the given kernel
    function with the specified range and kernel arguments. Its purpose is to
    collect the information surfaced by the host-raising process about a 
    particular launch, thereby providing an entry-point for host-device analyses
    and optimizations.

    If the `range` operand is not present, this operation represents a
    *single_task* launch, otherwise a *parallel_for* invocation. The `range`
    operand expects an `nd_range` pointer or a `range` pointer. In the latter
    case, an `id` pointer can be optionally given as the `offset`. In the former
    case, the `nd_range` attribute must be set.

    Note that due to the current focus on host-device optimizations for
    individual launches, this operation currently does not model the queue or
    the command-group handler associated with the launch, nor does it observe or
    yield any events.

    Example:
    ```
    sycl.host.schedule_kernel @kernels::@k0[range %1](%2, %3)
      : (!llvm.ptr, !llvm.ptr, i32) -> ()
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel_name,
    Optional<LLVM_AnyPointer>:$range,
    Optional<LLVM_AnyPointer>:$offset,
    Variadic<AnyType>:$args,
    UnitAttr:$nd_range);
  let builders = [
    OpBuilder<(ins "::mlir::SymbolRefAttr":$kernel_name,
                   "::mlir::ValueRange":$args), [{
      build($_builder, $_state, kernel_name, /*range=*/Value(),
            /*offset=*/Value(), args);
    }]>,
    OpBuilder<(ins "::mlir::SymbolRefAttr":$kernel_name,
                   "::mlir::Value":$range,
                   "::mlir::ValueRange":$args,
                   "bool":$nd_range), [{
      auto ndRangeAttr = nd_range ? $_builder.getUnitAttr() : UnitAttr();
      build($_builder, $_state, kernel_name, range,
            /*offset=*/Value(), args, ndRangeAttr);
    }]>,
    OpBuilder<(ins "::mlir::SymbolRefAttr":$kernel_name,
                   "::mlir::Value":$range,
                   "::mlir::Value":$offset,
                   "::mlir::ValueRange":$args), [{
      build($_builder, $_state, kernel_name, range, offset, args);
    }]>
  ];
  let assemblyFormat = [{
    $kernel_name (`[` (`nd_range` $nd_range^):(```range`)? $range^ (`,` `offset` $offset^)? `]`)? `` (`(` $args^ `)`)?
      attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // SYCL_HOST_OPS
