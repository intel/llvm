//===-- SYCLBase.td - SYCL dialect base definitions *- tablegen -*---------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base definition file for the SYCL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_BASE
#define SYCL_BASE

include "mlir/IR/OpBase.td"

////////////////////////////////////////////////////////////////////////////////
// DIALECT DECLARATION
////////////////////////////////////////////////////////////////////////////////

def SYCL_Dialect : Dialect {
  let name = "sycl";
  let cppNamespace = "::mlir::sycl";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    MethodRegistry methods;

    static constexpr llvm::StringRef getArgumentTypesAttrName() { return "ArgumentTypes"; }
    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }

    /// Each operation will be registered as usual in an MLIR dialect
    /// and also will be registered as a SYCL method if it implements one.
    template <typename... Args> void addOperations();

    /// Returns the name of the operation implementing the queried
    /// method, if present.
    ///
    /// For a method to be queried, it must have been registered
    /// first.
    llvm::Optional<llvm::StringRef> findMethod(::mlir::TypeID Type,
                                               llvm::StringRef Name) const;

    /// Returns the name of the operation implementing the queried
    /// method, if present.
    ///
    /// For a method to be queried, it must have been registered
    /// first.
    ///
    /// This function is similar to findMethod(), but also works for
    /// methods registered as members of classes derived from \p BaseType.
    llvm::Optional<llvm::StringRef>
    findMethodFromBaseClass(::mlir::TypeID BaseType,
                            llvm::StringRef Name) const;

    /// Add a definition for the given method to be used when lowering
    /// SYCLMethodOpInterface instances.
    void registerMethodDefinition(llvm::StringRef Name, mlir::func::FuncOp Func);

    /// Retrieve a function definition previously registered with
    /// registerMethodDefinition().
    llvm::Optional<mlir::func::FuncOp>
    lookupMethodDefinition(llvm::StringRef Name,
                           mlir::FunctionType Type) const;
  }];
}

////////////////////////////////////////////////////////////////////////////////
// Base declarations
////////////////////////////////////////////////////////////////////////////////

class SYCL_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::sycl";
}

#endif // SYCL_BASE
