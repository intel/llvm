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
  let summary = "Represents the member initialization of a SYCL object.";

  let description = [{
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

def SYCLHostGetKernelOp
    : SYCL_HostOp<"get_kernel",
        [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Defines a reference to a SYCL kernel, i.e., a `gpu.func` " #
                "with `kernel` attribute.";
  let arguments = (ins SymbolRefAttr:$kernel_name);
  let results = (outs LLVM_AnyPointer:$res);
  let assemblyFormat = [{
    $kernel_name attr-dict `:` type($res)
  }];
}

def SYCLHostHandlerSetKernel
    : SYCL_HostOp<"handler.set_kernel",
        [SYCLHostHandlerOp, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Assigns a kernel to a `sycl::handler`, thus pairing the " #
                "handler and the kernel being launched.";
  let arguments = (ins Arg<LLVM_AnyPointer, "The handler", [MemWrite]>:$handler,
                       SymbolRefAttr:$kernel_name);
  let assemblyFormat = [{
    $handler `->` $kernel_name attr-dict `:` type($handler)
  }];
}

def SYCLHostHandlerSetNDRange
    : SYCL_HostOp<"handler.set_nd_range", [SYCLHostHandlerOp]> {
  let summary = "Assigns an nd-range to a `sycl::handler`, setting the " #
                "nd-range of the kernel to be launched.";
  let description = [{
    The `range` operand expects an `nd_range` pointer or a `range` pointer. In
    the latter case, an `id` pointer can be optionally given as the `offset`. In
    the former case, the `nd_range` attribute must be set.
  }];
  let arguments = (ins Arg<LLVM_AnyPointer, "The handler", [MemWrite]>:$handler,
                       Arg<LLVM_AnyPointer, "The range", [MemRead]>:$range,
                       Arg<Optional<LLVM_AnyPointer>, "The offset", [MemRead]>:$offset,
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
    function object `lambda` at index `index`. If a special SYCL entity is
    captured (e.g. an acccessor), its type is stored in the `sycl_type`
    attribute. The op is created during the progressive raising towards the
    `sycl.host.schedule_kernel` op.

    Example:
    ```
    sycl.host.set_captured %lambda[1] = %scalar : !llvm.ptr, !llvm.ptr, i32
    sycl.host.set_captured %lambda[2] = %accessor 
      : !llvm.ptr, !llvm.ptr, !llvm.ptr (!sycl_accessor_1_f32_rw_gb)
    ```
  }];
  let arguments = (ins
    Arg<LLVM_AnyPointer, "The lambda", [MemWrite]>:$lambda,
    I64Attr:$index,
    Arg<AnyType, "The captured value", [MemRead]>:$value,
    OptionalAttr<TypeAttr>:$sycl_type
  );
  let assemblyFormat = [{
    $lambda `[` $index `]` `=` $value attr-dict 
      `:` type(operands) (` ` `(` $sycl_type^ `)`)?
  }];
  let hasVerifier = 1;
}

def SYCLHostScheduleKernel : SYCL_HostOp<"schedule_kernel",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     AttrSizedOperandSegments]> {
  let summary = "Schedules a SYCL kernel launch.";
  let description = [{
    This operation represents the scheduling of a launch of the given kernel
    function with the specified handler, range and kernel arguments. Its purpose
    is to collect the information surfaced by the host-raising process about a 
    particular launch, thereby providing an entry-point for host-device analyses
    and optimizations.

    If the `range` operand is not present, this operation represents a
    *single_task* launch, otherwise a *parallel_for* invocation. The `range`
    operand expects an `nd_range` pointer or a `range` pointer. In the latter
    case, an `id` pointer can be optionally given as the `offset`. In the former
    case, the `nd_range` attribute must be set.

    Pointer arguments can be annotated with the SYCL type of the entity they
    refer to, e.g. an accessor. Internally, this information is stored in the
    `sycl_types` type array attribute, sized to match the number of arguments.
    If no such type annotation is available for an argument, the `None` type
    shall be used as a placeholder.

    Note that due to the current focus on host-device optimizations for
    individual launches, this operation currently does not model the queue or
    the command-group handler associated with the launch, nor does it observe or
    yield any events.

    Example:
    ```
    sycl.host.schedule_kernel %handler -> @kernels::@k0[range %1]
      (%2: !sycl_accessor_1_f32_rw_gb, %3)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    ```
  }];

  let arguments = (ins
    LLVM_AnyPointer:$handler,
    SymbolRefAttr:$kernel_name,
    Optional<LLVM_AnyPointer>:$range,
    Optional<LLVM_AnyPointer>:$offset,
    Variadic<AnyType>:$args,
    TypeArrayAttr:$sycl_types,
    UnitAttr:$nd_range);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$handler,
                   "::mlir::SymbolRefAttr":$kernel_name,
                   "::mlir::ValueRange":$args), [{
      build($_builder, $_state, handler, kernel_name, /*range=*/Value(),
            /*offset=*/Value(), args);
    }]>,
    OpBuilder<(ins "::mlir::Value":$handler,
                   "::mlir::SymbolRefAttr":$kernel_name,
                   "::mlir::Value":$range,
                   "::mlir::ValueRange":$args,
                   "bool":$nd_range), [{
      ::llvm::SmallVector<::mlir::Type> noneTypes(args.size(), $_builder.getNoneType());
      auto syclTypes = $_builder.getTypeArrayAttr(noneTypes);
      auto ndRangeAttr = nd_range ? $_builder.getUnitAttr() : UnitAttr();
      build($_builder, $_state, handler, kernel_name, range,
            /*offset=*/Value(), args, syclTypes, ndRangeAttr);
    }]>,
    OpBuilder<(ins "::mlir::Value":$handler,
                    "::mlir::SymbolRefAttr":$kernel_name,
                   "::mlir::Value":$range,
                   "::mlir::Value":$offset,
                   "::mlir::ValueRange":$args), [{
      ::llvm::SmallVector<::mlir::Type> noneTypes(args.size(), $_builder.getNoneType());
      auto syclTypes = $_builder.getTypeArrayAttr(noneTypes);
      build($_builder, $_state, handler, kernel_name, range, offset, args, syclTypes);
    }]>
  ];
  let assemblyFormat = [{
    $handler `->` $kernel_name
    (`[` (`nd_range` $nd_range^):(```range`)? $range^ (`,` `offset` $offset^)? `]`)? ``
    custom<ArgsWithSYCLTypes>($args, $sycl_types)
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SYCLHostSubmitOp : SYCL_HostOp<"submit",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Submits a SYCL kernel.";
  let description = [{
    This operation represents the submission of a SYCL kernel
    (`sycl::queue::submit`). In addition to the function implementing the CGF,
    this operation receives the values captured (by reference) by this.

    Example:
    ```
    sycl.host.submit %queue([%buff_a, %buff_b]@cgf_impl) -> %event
      : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$c_g_f_name,
    Arg<LLVM_AnyPointer, "The queue", [MemWrite]>:$queue,
    Arg<LLVM_AnyPointer, "The event", [MemWrite]>:$event,
    Variadic<AnyType>:$args);
  let assemblyFormat = [{
    $queue `(` (`[` $args^ `]`)? `` $c_g_f_name `)` `->` $event
      attr-dict `:` type(operands)
  }];
}

#endif // SYCL_HOST_OPS
