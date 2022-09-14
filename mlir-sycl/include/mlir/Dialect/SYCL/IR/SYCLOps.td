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

////////////////////////////////////////////////////////////////////////////////
// TYPE DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCL_Dialect : Dialect {
  let name = "sycl";
  let cppNamespace = "::mlir::sycl";
  let useDefaultTypePrinterParser = 1;
}

class SYCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<SYCL_Dialect, mnemonic, traits>;

def SYCL_IDType : DialectType<SYCL_Dialect, CPred<"$_self.isa<IDType>()">, "sycl::id type">;
def SYCL_AccessorType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorType>()">, "sycl::accessor type">;
def SYCL_AccessorCommonType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorCommonType>()">, "sycl::accessor_common type">;
def SYCL_RangeType : DialectType<SYCL_Dialect, CPred<"$_self.isa<RangeType>()">, "sycl::range type">;
def SYCL_AccessorImplType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorImplDeviceType>()">, "sycl::accessor_impl_device type">;
def SYCL_ArrayType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ArrayType>()">, "sycl::array type">;
def SYCL_ItemType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ItemType>()">, "sycl::item type">;
def SYCL_ItemBaseType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ItemBaseType>()">, "sycl::item_base type">;
def SYCL_NdItemType : DialectType<SYCL_Dialect, CPred<"$_self.isa<NdItemType>()">, "sycl::nd_item type">;
def SYCL_GroupType : DialectType<SYCL_Dialect, CPred<"$_self.isa<GroupType>()">, "sycl::group type">;

def IDMemRef : MemRefOf<[SYCL_IDType]>;
def AccessorMemRef : MemRefOf<[SYCL_AccessorType]>;
def AccessorCommonMemRef : MemRefOf<[SYCL_AccessorCommonType]>;
def RangeMemRef : MemRefOf<[SYCL_RangeType]>;
def AccessorImplMemRef : MemRefOf<[SYCL_AccessorImplType]>;
def ArrayMemRef : MemRefOf<[SYCL_ArrayType]>;
def ItemMemRef : MemRefOf<[SYCL_ItemType]>;
def ItemBaseMemRef : MemRefOf<[SYCL_ItemBaseType]>;
def NDItemMemRef : MemRefOf<[SYCL_NdItemType]>;
def GroupMemRef : MemRefOf<[SYCL_GroupType]>;

def SYCLMemref : AnyTypeOf<[
  IDMemRef,
  AccessorMemRef,
  AccessorCommonMemRef,
  RangeMemRef,
  AccessorImplMemRef,
  ArrayMemRef,
  ItemMemRef,
  ItemBaseMemRef,
  NDItemMemRef,
  GroupMemRef
]>;
def IndexType : AnyTypeOf<[I32, I64, Index]>;

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

#endif // SYCL_OPS
