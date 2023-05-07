//===--- SYCLMethodOpInterface.td -----------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_METHOD_OP_INTERFACE
#define SYCL_METHOD_OP_INTERFACE

include "mlir/Dialect/SYCL/IR/SYCLBase.td"

include "mlir/IR/OpBase.td"

def SYCLMethodOpInterface : SYCL_OpInterface<"SYCLMethodOpInterface"> {
  let description =[{
    This interface describes an operation that calls a member function of a SYCL
    type.

    Operations implementing this interface should inherit
    `SYCLMethodOpInterfaceImpl`.

    Operations implementing this interface are registered as methods in the
    `SYCLDialect`. See the `SYCLDialect` documentation.
  }];

  let methods = [
    StaticInterfaceMethod<[{
      Return the ID of the type this method is a member of.
    }], "::mlir::TypeID", "getTypeID">,
    StaticInterfaceMethod<[{
      Return the list of the method names to be replaced by this
      operation.
    }], "llvm::ArrayRef<llvm::StringLiteral>", "getMethodNames">,
  ];
}

#endif // SYCL_METHOD_OP_INTERFACE
