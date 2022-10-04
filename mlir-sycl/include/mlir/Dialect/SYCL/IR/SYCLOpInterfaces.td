// Copyright (C) Intel

//===--- SYCLOpInterfaces.td ----------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_OP_INTERFACES
#define SYCL_OP_INTERFACES

include "mlir/IR/OpBase.td"

class SYCL_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::sycl";
}

def SYCLMethodOpInterface : SYCL_OpInterface<"SYCLMethodOpInterface"> {
  let description =[{
    This interface describes an operation that calls a member function of a SYCL
    type.

    Operations implementing this interface should inherit
    `SYCLMethodOpInterfaceImpl` and also implement the `MangledName` method,
    returning the mangled name of the function implementing this operation; the
    `BaseType` method, returning the SYCL type representing the class this
    method is a member of; the `Type` method, returning the original name of the
    type this operation is a member of; and the `Funcion` method, returning the
    name of the function implementing this operation.

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
    InterfaceMethod<[{
      Return the SYCL type this method is implemented in.
    }], "::mlir::Type", "getBaseType">,
    InterfaceMethod<[{
      Return the mangled name of the function implementing this operation.
    }], "llvm::StringRef", "getMangledFunctionName">,
    InterfaceMethod<[{
      Return the name of the function implementing this operation.
    }], "llvm::StringRef", "getFunctionName">,
    InterfaceMethod<[{
      Return the original name of the type this method is implemented in.
    }], "llvm::StringRef", "getType">,
  ];
}

#endif // SYCL_OP_INTERFACES
