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
  }];

  let extraClassDefinition = [{
  constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> $cppClass::methods;
  }];
}

def SYCL_IDType : DialectType<SYCL_Dialect, CPred<"$_self.isa<IDType>()">, "sycl::id type">;
def SYCL_AccessorCommonType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorCommonType>()">, "sycl::accessor_common type">;
def SYCL_AccessorType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorType>()">, "sycl::accessor type">;
def SYCL_RangeType : DialectType<SYCL_Dialect, CPred<"$_self.isa<RangeType>()">, "sycl::range type">;
def SYCL_NdRangeType : DialectType<SYCL_Dialect, CPred<"$_self.isa<NdRangeType>()">, "sycl::nd_range type">;
def SYCL_AccessorImplType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AccessorImplDeviceType>()">, "sycl::accessor_impl_device type">;
def SYCL_ArrayType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ArrayType>()">, "sycl::array type">;
def SYCL_ItemType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ItemType>()">, "sycl::item type">;
def SYCL_ItemBaseType : DialectType<SYCL_Dialect, CPred<"$_self.isa<ItemBaseType>()">, "sycl::item_base type">;
def SYCL_NdItemType : DialectType<SYCL_Dialect, CPred<"$_self.isa<NdItemType>()">, "sycl::nd_item type">;
def SYCL_GroupType : DialectType<SYCL_Dialect, CPred<"$_self.isa<GroupType>()">, "sycl::group type">;
def SYCL_AtomicType : DialectType<SYCL_Dialect, CPred<"$_self.isa<AtomicType>()">, "sycl::atomic type">;
def SYCL_MultiPtrType : DialectType<SYCL_Dialect, CPred<"$_self.isa<MultiPtrType>()">, "sycl::multi_ptr type">;


def IDMemRef : MemRefOf<[SYCL_IDType]>;
def AccessorCommonMemRef : MemRefOf<[SYCL_AccessorCommonType]>;
def AccessorMemRef : MemRefOf<[SYCL_AccessorType]>;
def RangeMemRef : MemRefOf<[SYCL_RangeType]>;
def NDRangeMemRef : MemRefOf<[SYCL_NdRangeType]>;
def AccessorImplMemRef : MemRefOf<[SYCL_AccessorImplType]>;
def ArrayMemRef : MemRefOf<[SYCL_ArrayType]>;
def ItemMemRef : MemRefOf<[SYCL_ItemType]>;
def ItemBaseMemRef : MemRefOf<[SYCL_ItemBaseType]>;
def NDItemMemRef : MemRefOf<[SYCL_NdItemType]>;
def GroupMemRef : MemRefOf<[SYCL_GroupType]>;
def AtomicMemRef : MemRefOf<[SYCL_AtomicType]>;
def MultiPtrMemRef : MemRefOf<[SYCL_MultiPtrType]>;


def SYCLMemref : AnyTypeOf<[
  IDMemRef,
  AccessorCommonMemRef,
  AccessorMemRef,
  RangeMemRef,
  NDRangeMemRef,
  AccessorImplMemRef,
  ArrayMemRef,
  ItemMemRef,
  ItemBaseMemRef,
  NDItemMemRef,
  GroupMemRef,
  AtomicMemRef,
  MultiPtrMemRef
]>;
def IndexType : AnyTypeOf<[I32, I64, Index]>;
def SYCLGetResult : AnyTypeOf<[I64, MemRefOf<[I64]>]>;
def SYCLGetIDResult : AnyTypeOf<[I64, SYCL_IDType]>;
def SYCLGetRangeResult : AnyTypeOf<[I64, SYCL_RangeType]>;

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
    FlatSymbolRefAttr:$MangledFunctionName,
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
  Pure, MemRefsNormalizable]
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
    OptionalAttr<FlatSymbolRefAttr>:$TypeName,
    FlatSymbolRefAttr:$FunctionName,
    FlatSymbolRefAttr:$MangledFunctionName,
    Variadic<AnyType>:$Args
  );
  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
    "::llvm::Optional<::mlir::Type>":$result,
    "::llvm::Optional<::llvm::StringRef>":$TypeName,
    "::llvm::StringRef":$FunctionName,
    "::llvm::StringRef":$MangledFunctionName,
    "::mlir::ValueRange":$Args), [{
      odsState.addOperands(Args);
      if (TypeName.has_value()) {
        odsState.addAttribute(getTypeNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), TypeName.value()));
      }
      odsState.addAttribute(getFunctionNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), FunctionName));
      odsState.addAttribute(getMangledFunctionNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), MangledFunctionName));
      if (result.has_value()) {
        odsState.addTypes(result.value());
      }
    }]>
  ];

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// accessor.subscript OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorSubscriptIndex : AnyTypeOf<[IndexType, IDMemRef]>;

def SYCLAccessorSubscriptOp
    : SYCLMethodOpInterfaceImpl<"accessor.subscript", "AccessorType",
                                ["operator[]"], [Pure]> {
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
                                ["get", "operator[]"],
                                [Pure, SYCLGetComponent]> {
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
}

////////////////////////////////////////////////////////////////////////////////
// range.size OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLRangeSizeOp
    : SYCLMethodOpInterfaceImpl<"range.size", "RangeType",
                                ["size"], [Pure]> {
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

////////////////////////////////////////////////////////////////////////////////
// nd_range.get_global_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNdRangeGetGlobalRange
    : SYCLMethodOpInterfaceImpl<"nd_range.get_global_range", "NdRangeType",
                                ["get_global_range"], [Pure]> {
  let summary = "Call to nd_range::get_global_range";
  let description = [{
    This operation represents a call to the nd_range::get_global_range function.
  }];

  let arguments = (ins NDRangeMemRef:$ND,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_range.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNdRangeGetLocalRange
    : SYCLMethodOpInterfaceImpl<"nd_range.get_local_range", "NdRangeType",
                                ["get_local_range"], [Pure]> {
  let summary = "Call to nd_range::get_local_range";
  let description = [{
    This operation represents a call to the nd_range::get_local_range function.
  }];

  let arguments = (ins NDRangeMemRef:$ND,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_range.get_group_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNdRangeGetGroupRange
    : SYCLMethodOpInterfaceImpl<"nd_range.get_group_range", "NdRangeType",
                                ["get_group_range"], [Pure]> {
  let summary = "Call to nd_range::get_group_range";
  let description = [{
    This operation represents a call to the nd_range::get_group_range function.
  }];

  let arguments = (ins NDRangeMemRef:$ND,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// id.get OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLIDGetOp
    : SYCLMethodOpInterfaceImpl<"id.get", "IDType",
                                ["get", "operator[]", "operator unsigned long"],
                                [Pure, SYCLGetComponent]> {
  let summary = "Call to id::get/operator[]/operator size_t";
  let description = [{
    This operation represents a call to the id::get/operator[]/operator size_t functions.
  }];

  let arguments = (ins IDMemRef:$ID,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// item.get_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLItemGetIDOp
    : SYCLMethodOpInterfaceImpl<"item.get_id", "ItemType",
                                ["get_id", "operator[]", "operator unsigned long"],
                                [Pure, SYCLGetID]> {
  let summary = "Call to item::get_id/operator[]/operator size_t";
  let description = [{
    This operation represents a call to the item::get_id/operator[]/operator size_t functions.
  }];

  let arguments = (ins ItemMemRef:$Item,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// item.get_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLItemGetRangeOp
    : SYCLMethodOpInterfaceImpl<"item.get_range", "ItemType",
                                ["get_range"], [Pure, SYCLGetRange]> {
  let summary = "Call to item::get_range";
  let description = [{
    This operation represents a call to the item::get_range function.
  }];

  let arguments = (ins ItemMemRef:$Item,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// item.get_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLItemGetLinearIDOp
    : SYCLMethodOpInterfaceImpl<"item.get_linear_id", "ItemType",
                                ["get_linear_id"], [Pure]> {
  let summary = "Call to item::get_linear_id";
  let description = [{
    This operation represents a call to the item::get_linear_id function.
  }];

  let arguments = (ins ItemMemRef:$Item,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_global_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGlobalIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_global_id", "NdItemType",
                                ["get_global_id"], [Pure, SYCLGetID]> {
  let summary = "Call to nd_item::get_global_id";
  let description = [{
    This operation represents a call to the nd_item::get_global_id function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_global_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGlobalLinearIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_global_linear_id", "NdItemType",
                                ["get_global_linear_id"], [Pure]> {
  let summary = "Call to nd_item::get_global_linear_id";
  let description = [{
    This operation represents a call to the nd_item::get_global_linear_id function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetLocalIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_local_id", "NdItemType",
                                ["get_local_id"], [Pure, SYCLGetID]> {
  let summary = "Call to nd_item::get_local_id";
  let description = [{
    This operation represents a call to the nd_item::get_local_id function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetLocalLinearIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_local_linear_id", "NdItemType",
                                ["get_local_linear_id"], [Pure]> {
  let summary = "Call to nd_item::get_local_linear_id";
  let description = [{
    This operation represents a call to the nd_item::get_local_linear_id function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_group OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGroupOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_group", "NdItemType",
                                ["get_group"], [Pure, SYCLGetGroup]> {
  let summary = "Call to nd_item::get_group";
  let description = [{
    This operation represents a call to the nd_item::get_group function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs AnyTypeOf<[I64, SYCL_GroupType]>:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_group_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGroupLinearIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_group_linear_id", "NdItemType",
                                ["get_group_linear_id"], [Pure]> {
  let summary = "Call to nd_item::get_group_linear_id";
  let description = [{
    This operation represents a call to the nd_item::get_group_linear_id function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_group_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGroupRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_group_range", "NdItemType",
                                ["get_group_range"],
                                [Pure, SYCLGetRange]> {
  let summary = "Call to nd_item::get_group_range";
  let description = [{
    This operation represents a call to the nd_item::get_group_range function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_global_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGlobalRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_global_range", "NdItemType",
                                ["get_global_range"],
                                [Pure, SYCLGetRange]> {
  let summary = "Call to nd_item::get_global_range";
  let description = [{
    This operation represents a call to the nd_item::get_global_range function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetLocalRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_local_range", "NdItemType",
                                ["get_local_range"],
                                [Pure, SYCLGetRange]> {
  let summary = "Call to nd_item::get_local_range";
  let description = [{
    This operation represents a call to the nd_item::get_local_range function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetNdRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_nd_range", "NdItemType",
                                ["get_nd_range"], [Pure]> {
  let summary = "Call to nd_item::get_nd_range";
  let description = [{
    This operation represents a call to the nd_item::get_nd_range function.
  }];

  let arguments = (ins NDItemMemRef:$NDItem,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCL_NdRangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetGroupIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_id", "GroupType",
                                ["get_group_id", "operator[]"],
                                [Pure, SYCLGetID]> {
  let summary = "Call to group::get_group_id/operator[]";
  let description = [{
    This operation represents a call to the group::get_group_id/operator[] functions.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_id", "GroupType",
                                ["get_local_id"],
                                [Pure, SYCLGetID]> {
  let summary = "Call to group::get_local_id";
  let description = [{
    This operation represents a call to the group::get_local_id function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_range", "GroupType",
                                ["get_local_range"],
                                [Pure, SYCLGetRange]> {
  let summary = "Call to group::get_local_range";
  let description = [{
    This operation represents a call to the group::get_local_range function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetGroupRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_range", "GroupType",
                                ["get_group_range"],
                                [Pure, SYCLGetRange]> {
  let summary = "Call to group::get_group_range";
  let description = [{
    This operation represents a call to the group::get_group_range function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       Optional<I32>:$Index,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_max_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetMaxLocalRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_max_local_range", "GroupType",
                                ["get_max_local_range"], [Pure]> {
  let summary = "Call to group::get_max_local_range";
  let description = [{
    This operation represents a call to the group::get_max_local_range function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetGroupLinearIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_linear_id", "GroupType",
                                ["get_group_linear_id"], [Pure]> {
  let summary = "Call to group::get_group_linear_id";
  let description = [{
    This operation represents a call to the group::get_group_linear_id function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalLinearIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_linear_id", "GroupType",
                                ["get_local_linear_id"], [Pure]> {
  let summary = "Call to group::get_local_linear_id";
  let description = [{
    This operation represents a call to the group::get_local_linear_id function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_linear_range OPERATION
////////////////////////////////////////////////////////////////////////////////

 def SYCLGroupGetGroupLinearRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_linear_range", "GroupType",
                                ["get_group_linear_range"], [Pure]> {
  let summary = "Call to group::get_group_linear_range";
  let description = [{
    This operation represents a call to the group::get_group_linear_range function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_linear_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalLinearRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_linear_range", "GroupType",
                                ["get_local_linear_range"], [Pure]> {
  let summary = "Call to group::get_local_linear_range";
  let description = [{
    This operation represents a call to the group::get_local_linear_range function.
  }];

  let arguments = (ins GroupMemRef:$Group,
                       TypeAttr:$BaseType,
                       FlatSymbolRefAttr:$FunctionName,
                       FlatSymbolRefAttr:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

#endif // SYCL_OPS
