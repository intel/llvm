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
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    MethodRegistry methods;

    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }

    /// In the example below, %p0 and %p1 are not aliased.
    /// void func(%arg0 : memref<?x!sycl.accessor> {sycl.inner.disjoint},
    ///           %arg1 : memref<?x!sycl.accessor> {sycl.inner.disjoint}) {
    ///   %p0 = sycl.accessor.subscript %arg0(%id0)
    ///   %p1 = sycl.accessor.subscript %arg1(%id1)
    /// }
    static StringLiteral getInnerDisjointAttrName() { return "sycl.inner.disjoint"; }

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
  }];
  let usePropertiesForAttributes = 1;
}

////////////////////////////////////////////////////////////////////////////////
// Base declarations
////////////////////////////////////////////////////////////////////////////////

class SYCL_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::sycl";
}

#endif // SYCL_BASE
