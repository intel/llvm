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
    `SYCLMethodOpInterfaceImpl` and also implement the `ArgumentTypes` method,
    returning the function argument types to be passed to the function
    implementing this operation; the `MangledFunctionName` method, returning the
    mangled name of the function implementing this operation; the `FunctionName`
    method, returning the unmangled name of the function implementing this
    operation; and the `TypeName` method, returning the original name of the
    type this operation is a member of.

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
      Return the SYCL type this method is a member of.
    }], "::mlir::Type", "getBaseType", /*args=*/ (ins),
    /*methodBody=*/ [{
      return $_op.getArgumentTypes()[0].template cast<::mlir::TypeAttr>().getValue();
    }]
    >,
    InterfaceMethod<[{
      Returns the argument types of the function implementing this operation.
    }], "::mlir::SmallVector<::mlir::Type>", "getArgumentTypes",
    /*args=*/ (ins), /*methodBody=*/ [{
      SmallVector<Type> Res;
      const auto Tys =
        $_op.getArgumentTypes().template getAsRange<::mlir::TypeAttr>();
      std::transform(Tys.begin(), Tys.end(), std::back_inserter(Res),
                     [](auto Attr) { return Attr.getValue(); });
      return Res;
    }]
    >,
    InterfaceMethod<[{
      Return the mangled name of the function implementing this operation.
    }], "llvm::Optional<llvm::StringRef>", "getMangledFunctionName">,
    InterfaceMethod<[{
      Return the name of the function implementing this operation.
    }], "llvm::StringRef", "getFunctionName">,
    InterfaceMethod<[{
      Return the original name of the type this method is implemented in.
    }], "llvm::StringRef", "getTypeName">
  ];
  // We cannot easily verify other attributes.
  let verify = [{
    if (static_cast<SYCLMethodOpInterface>($_op).getArgumentTypes().size() !=
          $_op->getNumOperands()) {
      return $_op->emitOpError("mismatch between number of arguments and input types");
    }
    auto interface = cast<mlir::sycl::SYCLMethodOpInterface>(op);
    const llvm::ArrayRef<llvm::StringLiteral> methods = interface.getMethodNames();
    const StringRef functionName = interface.getFunctionName();
    const auto iter = std::find(methods.begin(), methods.end(), functionName);
    if (iter == methods.end()) {
      return $_op->emitOpError("invalid function name for this operation: ") << functionName;
    }
    return mlir::success();
  }];
}

class SYCLOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::sycl";
}

def SYCLGetID : SYCLOpTrait<"SYCLGetID">;
def SYCLGetComponent : SYCLOpTrait<"SYCLGetComponent">;
def SYCLGetRange : SYCLOpTrait<"SYCLGetRange">;
def SYCLGetGroup : SYCLOpTrait<"SYCLGetGroup">;

#endif // SYCL_OP_INTERFACES
