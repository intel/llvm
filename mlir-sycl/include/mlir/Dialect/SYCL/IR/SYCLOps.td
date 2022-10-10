// Copyright (C) Codeplay Software Limited

//===--- SYCLOps.td -------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_OPS
#define SYCL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SYCLOpInterfaces.td"

////////////////////////////////////////////////////////////////////////////////
// TYPE DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCL_Dialect : Dialect {
  let name = "sycl";
  let cppNamespace = "::mlir::sycl";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    MethodRegistry methods;

    static constexpr llvm::StringRef getBaseTypeAttrName() { return "BaseType"; }
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
  }];
}

class SYCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<SYCL_Dialect, mnemonic, traits>;

// methodNames MUST be a sorted list as the interface verification uses binary
// search to verify FunctionName.
class SYCLMethodOpInterfaceImpl<
    string mnemonic, string type, list<string> methodNames, list<Trait> traits = []>
        : SYCL_Op<mnemonic, !listconcat(traits, [SYCLMethodOpInterface])> {
  string baseType = type;
  list<string> memberFunctionNames = methodNames;
  int arrSize = !size(memberFunctionNames);

  let extraClassDeclaration = [{
    static constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> methods{{ }] #
      !interleave(!foreach(name, memberFunctionNames, "\"" # name # "\""), ", ") # [{
     }};
    static ::mlir::TypeID getTypeID() { return ::mlir::sycl::}] # type # [{::getTypeID(); }
    static constexpr llvm::ArrayRef<llvm::StringLiteral> getMethodNames() { return methods; }
    ::mlir::Type getBaseType() { return BaseType(); }
    llvm::StringRef getMangledFunctionName() { return MangledFunctionName(); }
    llvm::StringRef getFunctionName() { return FunctionName(); }
    llvm::StringRef getTypeName() { return TypeName(); }
  }];

  let extraClassDefinition = [{
  constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> $cppClass::methods;
  }];
}

def SYCL_IDType : DialectType<SYCL_Dialect, CPred<"$_self.isa<IDType>()">, "sycl::id type">;
def SYCL_AccessorCommonType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorCommonType>()">, "sycl::accessor_common type">;
def SYCL_AccessorType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorType>()">, "sycl::accessor type">;
def SYCL_RangeType : DialectType<SYCL_Dialect, CPred<"$_self.isa<RangeType>()">, "sycl::range type">;
def SYCL_AccessorImplType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorImplDeviceType>()">, "sycl::accessor_impl_device type">;
def SYCL_ArrayType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ArrayType>()">, "sycl::array type">;
def SYCL_ItemType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ItemType>()">, "sycl::item type">;
def SYCL_ItemBaseType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ItemBaseType>()">, "sycl::item_base type">;
def SYCL_NdItemType : DialectType<SYCL_Dialect, CPred<"$_self.isa<NdItemType>()">, "sycl::nd_item type">;
def SYCL_GroupType : DialectType<SYCL_Dialect, CPred<"$_self.isa<GroupType>()">, "sycl::group type">;

def IDMemRef : MemRefOf<[SYCL_IDType]>;
def AccessorCommonMemRef : MemRefOf<[SYCL_AccessorCommonType]>;
def AccessorMemRef : MemRefOf<[SYCL_AccessorType]>;
def RangeMemRef : MemRefOf<[SYCL_RangeType]>;
def AccessorImplMemRef : MemRefOf<[SYCL_AccessorImplType]>;
def ArrayMemRef : MemRefOf<[SYCL_ArrayType]>;
def ItemMemRef : MemRefOf<[SYCL_ItemType]>;
def ItemBaseMemRef : MemRefOf<[SYCL_ItemBaseType]>;
def NDItemMemRef : MemRefOf<[SYCL_NdItemType]>;
def GroupMemRef : MemRefOf<[SYCL_GroupType]>;

def SYCLMemref : AnyTypeOf<[
  IDMemRef,
  AccessorCommonMemRef,
  AccessorMemRef,
  RangeMemRef,
  AccessorImplMemRef,
  ArrayMemRef,
  ItemMemRef,
  ItemBaseMemRef,
  NDItemMemRef,
  GroupMemRef
]>;
def IndexType : AnyTypeOf<[I32, I64, Index]>;
def SYCLGetResult : AnyTypeOf<[I64, MemRefOf<[I64]>]>;

////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR OPERATION
////////////////////////////////////////////////////////////////////////////////

def ConstructorArgs : AnyTypeOf<[SYCLMemref, IndexType, SYCL_IDType, SYCL_RangeType]>;
def SYCLConstructorOp : SYCL_Op<"constructor", []> {
  let summary = "Generic constructor operation";
  let description = [{
    This operation represent the call to the constructor of a SYCL type.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$Type,
    FlatSymbolRefAttr:$MangledName,
    Variadic<ConstructorArgs>:$Args
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// CAST OPERATION
////////////////////////////////////////////////////////////////////////////////

def CastArg : AnyTypeOf<[IDMemRef, RangeMemRef]>;
def CastRes : AnyTypeOf<[ArrayMemRef]>;
def SYCLCastOp : SYCL_Op<"cast", [DeclareOpInterfaceMethods<CastOpInterface>, 
  NoSideEffect, MemRefsNormalizable]
> {
  let summary = "Derive to base cast operation";
  let description = [{
    This operation is used when a method of the base type is called with its
    derived type like sycl::id, In this case the sycl::cast operation will
    cast the type to its base type.
  }];

  let arguments = (ins CastArg:$source);
  let results = (outs CastRes:$result);

  let assemblyFormat = [{
    `(` $source `)` attr-dict `:` functional-type($source, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// CALL OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLCallOp : SYCL_Op<"call", []> {
  let summary = "Generic call operation";
  let description = [{
    This operation represent the call to any function part of the sycl's
    namespace.
  }];

  let arguments = (ins
    OptionalAttr<FlatSymbolRefAttr>:$Type,
    FlatSymbolRefAttr:$Function,
    FlatSymbolRefAttr:$MangledName,
    Variadic<AnyType>:$Args
  );
  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
    "::llvm::Optional<::mlir::Type>":$result,
    "::llvm::Optional<::llvm::StringRef>":$Type,
    "::llvm::StringRef":$Function,
    "::llvm::StringRef":$MangledName,
    "::mlir::ValueRange":$Args), [{
      odsState.addOperands(Args);
      if (Type.hasValue()) {
        odsState.addAttribute(TypeAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), Type.getValue()));
      }
      odsState.addAttribute(FunctionAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), Function));
      odsState.addAttribute(MangledNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), MangledName));
      if (result.hasValue()) {
        odsState.addTypes(result.getValue());
      }
    }]>
  ];

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// accessor.getelementptr OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorSubscriptIndex : AnyTypeOf<[IndexType, SYCL_IDType]>;

def SYCLAccessorSubscriptOp
    : SYCLMethodOpInterfaceImpl<"accessor.subscript", "AccessorType",
                                ["operator[]"], [NoSideEffect]> {
  let summary = "Call to accessor::operator[]";
  let description = [{
    This operation represents a call to the accessor::operator[] function.
  }];

  let arguments = (ins AccessorMemRef:$Acc,
                       SYCLAccessorSubscriptIndex:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs AnyType:$Res);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $Acc `[` $Index `]` attr-dict `:` functional-type(operands, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// range.get OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLRangeGetOp
    : SYCLMethodOpInterfaceImpl<"range.get", "RangeType",
                                ["get", "operator[]"], [NoSideEffect]> {
  let summary = "Call to range::get/operator[]";
  let description = [{
    This operation represents a call to the range::get/operator[] functions.
  }];

  let arguments = (ins RangeMemRef:$Range,
                       I32:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetResult:$Res);

  let hasVerifier = 1;
}

////////////////////////////////////////////////////////////////////////////////
// range.size OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLRangeSizeOp
    : SYCLMethodOpInterfaceImpl<"range.size", "RangeType",
                                ["size"], [NoSideEffect]> {
  let summary = "Call to range::size[]";
  let description = [{
    This operation represents a call to the range::size[] function.
  }];

  let arguments = (ins RangeMemRef:$Range,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

#endif // SYCL_OPS
