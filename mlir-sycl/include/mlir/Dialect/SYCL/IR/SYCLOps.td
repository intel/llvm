//===-- SYCLOps.td - SYCL dialect operation definitions *- tablegen -*-----===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for SYCL dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_OPS
#define SYCL_OPS

include "mlir/IR/OpBase.td"

include "mlir/Dialect/SYCL/IR/SYCLBase.td"
include "mlir/Dialect/SYCL/IR/SYCLTypes.td"
include "mlir/Dialect/SYCL/IR/SYCLAttributes.td"
include "mlir/Dialect/SYCL/IR/SYCLMethodOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

////////////////////////////////////////////////////////////////////////////////
// BASE DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

class SYCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<SYCL_Dialect, mnemonic, traits>;

class SYCLOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::sycl";
}

class SYCLMethodOpInterfaceImpl<
    string mnemonic, string type, list<string> methodNames, list<Trait> traits = []>
        : SYCL_Op<mnemonic, !listconcat(traits, [SYCLMethodOpInterface])> {
  string baseType = type;
  list<string> memberFunctionNames = methodNames;
  int arrSize = !size(memberFunctionNames);

  code extraClassDeclarationBase = [{
    static constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> methods{{ }] #
      !interleave(!foreach(name, memberFunctionNames, "\"" # name # "\""), ", ") # [{
     }};
    static ::mlir::TypeID getTypeID() { return ::mlir::sycl::}] # type # [{::getTypeID(); }
    static constexpr llvm::ArrayRef<llvm::StringLiteral> getMethodNames() { return methods; }
  }];

  let extraClassDefinition = [{
    constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> $cppClass::methods;
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class SYCLConstructor<string name, list<Trait> traits = []>
    : SYCL_Op<name # ".constructor", traits # [MemoryEffectsOpInterface]> {
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// TRAIT DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCLGetID : SYCLOpTrait<"SYCLGetID">;
def SYCLGetComponent : SYCLOpTrait<"SYCLGetComponent">;
def SYCLGetRange : SYCLOpTrait<"SYCLGetRange">;
def SYCLGetGroup : SYCLOpTrait<"SYCLGetGroup">;
def SYCLIndexSpaceGetID : SYCLOpTrait<"SYCLIndexSpaceGetID">;
def SYCLIndexSpaceGetRange : SYCLOpTrait<"SYCLIndexSpaceGetRange">;

////////////////////////////////////////////////////////////////////////////////
// TYPES DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

// Define MemRef types (alphabetical order).
def AccessorCommonMemRef : MemRefOf<[SYCL_AccessorCommonType]>;
def AccessorImplDeviceMemRef : MemRefOf<[SYCL_AccessorImplDeviceType]>;
def AccessorMemRef : MemRefOf<[SYCL_AccessorType]>;
def AccessorSubscriptMemRef : MemRefOf<[SYCL_AccessorSubscriptType]>;
def ArrayMemRef : MemRefOf<[SYCL_ArrayType]>;
def AtomicMemRef : MemRefOf<[SYCL_AtomicType]>;
def GroupMemRef : MemRefOf<[SYCL_GroupType]>;
def HalfMemRef : MemRefOf<[SYCL_HalfType]>;
def IDMemRef : MemRefOf<[SYCL_IDType]>;
def ItemBaseMemRef : MemRefOf<[SYCL_ItemBaseType]>;
def ItemMemRef : MemRefOf<[SYCL_ItemType]>;
def LocalAccessorBaseDeviceMemRef : MemRefOf<[SYCL_LocalAccessorBaseDeviceType]>;
def LocalAccessorBaseMemRef : MemRefOf<[SYCL_LocalAccessorBaseType]>;
def LocalAccessorMemRef : MemRefOf<[SYCL_LocalAccessorType]>;
def MaximumMemRef : MemRefOf<[SYCL_MaximumType]>;
def MinimumMemRef : MemRefOf<[SYCL_MinimumType]>;
def MultiPtrMemRef : MemRefOf<[SYCL_MultiPtrType]>;
def NDItemMemRef : MemRefOf<[SYCL_NdItemType]>;
def NDRangeMemRef : MemRefOf<[SYCL_NdRangeType]>;
def OwnerLessBaseMemRef : MemRefOf<[SYCL_OwnerLessBaseType]>;
def RangeMemRef : MemRefOf<[SYCL_RangeType]>;
def StreamMemRef : MemRefOf<[SYCL_StreamType]>;
def SubGroupMemRef : MemRefOf<[SYCL_SubGroupType]>;
def SwizzledVecMemRef : MemRefOf<[SYCL_SwizzledVecType]>;
def VecMemRef : MemRefOf<[SYCL_VecType]>;

def SYCLMemref : AnyTypeOf<[
  // Keep in alphabetical order.
  AccessorCommonMemRef,
  AccessorImplDeviceMemRef,
  AccessorMemRef,
  AccessorSubscriptMemRef,
  ArrayMemRef,
  AtomicMemRef,
  GroupMemRef,  
  HalfMemRef,
  IDMemRef,
  ItemBaseMemRef,
  ItemMemRef,
  LocalAccessorBaseDeviceMemRef,
  LocalAccessorBaseMemRef,
  LocalAccessorMemRef,
  MaximumMemRef,
  MinimumMemRef,
  MultiPtrMemRef,
  NDItemMemRef,
  NDRangeMemRef,
  OwnerLessBaseMemRef,
  RangeMemRef,
  StreamMemRef,
  SubGroupMemRef,
  SwizzledVecMemRef,
  VecMemRef
]>;

// Other types (alphabetical order).
def IndexType : AnyTypeOf<[I32, I64, Index]>;
def SYCLGetIDResult : AnyTypeOf<[I64, SYCL_IDType]>;
def SYCLGetRangeResult : AnyTypeOf<[I64, SYCL_RangeType]>;
def SYCLGetResult : AnyTypeOf<[IndexType, MemRefOf<[IndexType]>]>;
def VectorEltTy : AnyTypeOf<[I1, I8, I16, I32, I64, F16, F32, F64]>;
def VectorSplatArg : MemRefOf<[VectorEltTy]>;

////////////////////////////////////////////////////////////////////////////////
// SYCL GRID OPERATIONS
////////////////////////////////////////////////////////////////////////////////

class SYCL1DGridOp<string mnemonic, list<Trait> traits = []>
    : SYCL_Op<mnemonic, !listconcat(traits, [Pure])> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class SYCL3DGridOp<string mnemonic, list<Trait> traits = []>
    : SYCL_Op<mnemonic, !listconcat(traits, [Pure])> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class SYCL3DIDGridOp<string mnemonic, list<Trait> traits = []> 
    : SYCL3DGridOp<mnemonic, traits> {
  let results = (outs SYCL_IDType:$result);
}

class SYCL3DRangeGridOp<string mnemonic, list<Trait> traits = []> 
    : SYCL3DGridOp<mnemonic, traits> {
  let results = (outs SYCL_RangeType:$result);
}

def SYCLNumWorkItemsOp : SYCL3DRangeGridOp<"num_work_items"> {
  let summary = "Retrieve the number of work-items.";
  let description = [{
    This operation returns the number of work-items in the index space.
  }];
}

def SYCLGlobalIDOp : SYCL3DIDGridOp<"global_id", [ResultsNonUniform]> {
  let summary = "Retrieve the global ID of the item.";
  let description = [{
    This operation returns the global ID of the item in the index space.
  }];
}

def SYCLLocalIDOp : SYCL3DIDGridOp<"local_id", [ResultsNonUniform]> {
  let summary = "Retrieve the local ID of the item.";
  let description = [{
    This operation returns the local ID of the item within the work-group.
  }];
}

def SYCLGlobalOffsetOp : SYCL3DIDGridOp<"global_offset">,
                       Deprecated<"deprecated in SYCL 2020"> {
  let summary = "Retrieve the global offset of the item.";
  let description = [{
    This operation returns the global offset of the item in the index space.

    Note that this is deprecated in SYCL 2020.
  }];
}

def SYCLNumWorkGroupsOp : SYCL3DRangeGridOp<"num_work_groups"> {
  let summary = "Retrieve the number of work-groups.";
  let description = [{
    This operation returns the number of work-groups in the index space.
  }];
}

def SYCLWorkGroupSizeOp : SYCL3DRangeGridOp<"work_group_size"> {
  let summary = "Retrieve the number of work-items in a work-group.";
  let description = [{
    This operation returns the number of work-items per work-group.
  }];
}

def SYCLWorkGroupIDOp : SYCL3DIDGridOp<"work_group_id", [ResultsNonUniform]> {
  let summary = "Retrieve the ID of the work-group.";
  let description = [{
    This operation returns the ID of the work-group.
  }];
}

def SYCLNumSubGroupsOp : SYCL1DGridOp<"num_sub_groups"> {
  let summary = "Retrieve the number of sub-groups.";
  let description = [{
    This operation returns the number of sub-groups.
  }];
}

def SYCLSubGroupMaxSizeOp : SYCL1DGridOp<"sub_group_max_size"> {
  let summary = "Retrieve the maximum size of a sub-group.";
  let description = [{
    This operation returns the maximum size of a sub-group.
  }];
}

def SYCLSubGroupSizeOp : SYCL1DGridOp<"sub_group_size"> {
  let summary = "Retrieve the sub-group size.";
  let description = [{
    This operation returns the sub-group size.
  }];
}

def SYCLSubGroupIDOp : SYCL1DGridOp<"sub_group_id", [ResultsNonUniform]> {
  let summary = "Retrieve the ID of the sub-group.";
  let description = [{
    This operation returns the ID of the sub-group.
  }];
}

def SYCLSubGroupLocalIDOp : SYCL1DGridOp<"sub_group_local_id", [ResultsNonUniform]> {
  let summary = "Retrieve the local ID of the sub-group.";
  let description = [{
    This operation returns the local ID of the sub-group.
  }];
}

////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLConstructorOp : SYCL_Op<"constructor",
    [MemoryEffectsOpInterface]> {
  let summary = "Generic constructor operation";
  let description = [{
    This operation represent the call to the constructor of a SYCL type.
  }];

  let arguments = (ins
    Arg<SYCLMemref, "The SYCL object being created">:$Dst,
    Arg<Variadic<AnyType>, "The constructor arguments">:$Args,
    FlatSymbolRefAttr:$TypeName,
    FlatSymbolRefAttr:$MangledFunctionName
  );
  let results = (outs);

  let extraClassDeclaration = [{
    /// Return the memory effects associated to each argument.
    ///
    /// The first argument, `this` will have `MemoryEffects::Write` memory
    /// effect associated and the rest of argument will have
    /// `MemoryEffects::Read` if they're a pointer/memref and no memory effect
    /// otherwise.
    void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                        ::mlir::MemoryEffects::Effect>> &effects);

    /// Returns the index of the SYCL object being created.
    unsigned getOutputOperandIndex() const { return 0; }
  }];

  let assemblyFormat = [{
    $TypeName `(` $Dst (`,` $Args^)?`)` attr-dict `:` `(` type($Dst) (`,` type($Args)^)? `)`
  }];
}

////////////////////////////////////////////////////////////////////////////////
// CAST OPERATION
////////////////////////////////////////////////////////////////////////////////

def CastArg : AnyTypeOf<[IDMemRef, RangeMemRef, AccessorMemRef, LocalAccessorMemRef, LocalAccessorBaseMemRef]>;
def CastRes : AnyTypeOf<[ArrayMemRef, AccessorCommonMemRef, LocalAccessorBaseMemRef, OwnerLessBaseMemRef]>;
def SYCLCastOp : SYCL_Op<"cast", [DeclareOpInterfaceMethods<CastOpInterface>, 
  Pure, ViewLikeOpInterface, MemRefsNormalizable]
> {
  let summary = "Derived type to base type cast operation";
  let description = [{
    This operation is used when a method of the base type is called with its
    derived type like sycl::id, In this case the sycl::cast operation will
    cast the type to its base type.
  }];

  let arguments = (ins CastArg:$source);
  let results = (outs CastRes:$result);

  let extraClassDeclaration = [{
    Value getViewSource() { return getSource(); }
  }];  

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];
}

def SYCLAddrSpaceCastOp : SYCL_Op<"addrspacecast", [
  DeclareOpInterfaceMethods<CastOpInterface>, 
  ViewLikeOpInterface, Pure
]> {
  let summary = "Perform an address space cast between specific and generic address space";

  let arguments = (ins AnyMemRef : $source);
  let results = (outs AnyMemRef : $result);

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];
}


////////////////////////////////////////////////////////////////////////////////
// CALL OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLCallOp : SYCL_Op<"call", [CallOpInterface]> {
  let summary = "Generic call operation";
  let description = [{
    This operation represent a call to any function part of the sycl's namespace.
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

  let extraClassDeclaration = [{
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of the generic SYCL call operation, this is required by
    /// the call interface.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<FlatSymbolRefAttr>(getMangledFunctionNameAttrName());
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr(getMangledFunctionNameAttrName(), callee.get<SymbolRefAttr>());
    }

    /// Get the argument operands to the called function, this is required by the
    /// call interface.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    /// Get the argument operands to the called function as a mutable range, this is
    /// required by the call interface.
    MutableOperandRange getArgOperandsMutable() {
      return getArgsMutable();
    }
  }];    

  let assemblyFormat = [{
    $FunctionName `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// accessor.get_pointer OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorGetPointerOp
    : SYCLMethodOpInterfaceImpl<"accessor.get_pointer", "AccessorType",
                                ["get_pointer"]> {
  let summary = "Represents the accessor get_pointer operation";
  let description = [{
    Returns a pointer to the start of this accessor's memory.
  }];

  let arguments = (ins Arg<AccessorMemRef, "The accessor", [MemRead]>:$Acc);

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasVerifier = 1;

  let results = (outs AnyMemRef:$result);
}

////////////////////////////////////////////////////////////////////////////////
// accessor.range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorGetRangeOp
    : SYCLMethodOpInterfaceImpl<"accessor.get_range", "AccessorType",
                                ["get_range"]> {
  let summary = "Represents the accessor get_range operation";
  let description = [{
    Returns a range object which represents the number of elements per dimension
    that this accessor may access.
  }];

  let arguments = (ins Arg<AccessorMemRef, "The accessor", [MemRead]>:$Acc);

  let extraClassDeclaration = extraClassDeclarationBase;  

  let hasVerifier = 1;

  let results = (outs SYCL_RangeType:$result);
}

////////////////////////////////////////////////////////////////////////////////
// accessor.size OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorSizeOp
    : SYCLMethodOpInterfaceImpl<"accessor.size", "AccessorType",
                                ["size"]> {
  let summary = "Represents the accessor size operation";
  let description = [{
    Returns the number of elements of the memory region this accessor may
    access.
  }];

  let arguments = (ins Arg<AccessorMemRef, "The accessor", [MemRead]>:$Acc);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs IndexType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// accessor.subscript OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorSubscriptIndex : AnyTypeOf<[IndexType, IDMemRef]>;

def SYCLAccessorSubscriptOp
    : SYCLMethodOpInterfaceImpl<"accessor.subscript", "AccessorType",
                                ["operator[]"]> {
  let summary = "Call to accessor::operator[]";
  let description = [{
    This operation represents a call to the accessor::operator[] function.
  }];

  let arguments = (ins Arg<AccessorMemRef, "The accessor", [MemRead]>:$Acc,
                       Arg<SYCLAccessorSubscriptIndex, "The offset", [MemRead]>:$Index);
  let results = (outs AnyType:$Res);

  let extraClassDeclaration =  extraClassDeclarationBase # [{
    /// Returns the index of the accessor operand.
    unsigned getMemRefOperandIndex() const { return 0; }

    /// Returns the index of the offset operand.
    unsigned getOffsetOperandIndex() const { return 1; }
  }];

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
                                [SYCLGetComponent]> {
  let summary = "Call to range::get/operator[]";
  let description = [{
    This operation represents a call to the range::get/operator[] functions.
  }];

  let arguments = (ins Arg<RangeMemRef, "The input range", [MemRead]>:$Range,
                       I32:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetResult:$Res);

  let assemblyFormat = [{
    $Range `[` $Index `]` attr-dict `:` functional-type(operands, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// range.size OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLRangeSizeOp
    : SYCLMethodOpInterfaceImpl<"range.size", "RangeType",
                                ["size"]> {
  let summary = "Call to range::size[]";
  let description = [{
    This operation represents a call to the range::size[] function.
  }];

  let arguments = (ins Arg<RangeMemRef, "The input range", [MemRead]>:$Range);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// range.constructor OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLRangeConstructorOp : SYCLConstructor<"range"> {
  let summary = "Operation returning a new sycl.range value";
  let description = [{
    This operation creates a new `sycl.range`. The following signatures are
    provided:
    ```mlir
    # Initialize each member
    (index) -> memref<1x!sycl_range_1>
    (index, index) -> memref<1x!sycl_range_2>
    (index, index, index) -> memref<1x!sycl_range_3>
    # Copy constructor
    (memref<?xsycl_range_N>) -> memref<1x!sycl_range_N>
    ```
    In addition of initializing the values, this operation also allocates the
    object.
  }];
  let arguments = (ins Arg<Variadic<AnyType>, "Constructor arguments">:$Args);
  let results =
      (outs Res<MemRefRankOf<[SYCL_RangeType], [1]>, "constructed object">:$range);
  let extraClassDeclaration = [{
    /// Return the memory effects associated with each argument.
    ///
    /// Each argument will have `MemoryEffects::Read` if they're a
    /// pointer/memref and no memory effect otherwise.
    void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                        ::mlir::MemoryEffects::Effect>> &effects);
  }];
  let hasVerifier = true;
}

////////////////////////////////////////////////////////////////////////////////
// nd_range.constructor OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDRangeConstructorOp : SYCLConstructor<"nd_range"> {
  let summary = "Operation returning a new sycl.nd_range value";
  let description = [{
    This operation creates a new `sycl.nd_range`. The following signatures are
    provided:
    ```mlir
    # Initialize each member
    (memref<?xsycl_range_N>, memref<?xsycl_range_N>) -> memref<1x!sycl_nd_range_N>
    (memref<?xsycl_range_N>, memref<?xsycl_range_N>, memref<?xsycl_id_N>) -> memref<1x!sycl_nd_range_N>
    # Copy constructor
    (memref<?xsycl_nd_range_N>) -> memref<1x!sycl_nd_range_N>
    ```
    In addition of initializing the values, this operation also allocates the
    object.
  }];
  let arguments = (ins Arg<Variadic<AnyType>, "Constructor arguments">:$Args);
  let results =
      (outs Res<MemRefRankOf<[SYCL_NdRangeType], [1]>, "constructed object">:$NDRange);
  let extraClassDeclaration = [{
    /// Return the memory effects associated to each argument.
    ///
    /// Each argument will have `MemoryEffects::Read`.
    void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                        ::mlir::MemoryEffects::Effect>> &effects);
  }];
  let hasVerifier = true;
  let hasCanonicalizer = true;
}

////////////////////////////////////////////////////////////////////////////////
// nd_range.get_global_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNdRangeGetGlobalRange
    : SYCLMethodOpInterfaceImpl<"nd_range.get_global_range", "NdRangeType",
                                ["get_global_range"]> {
  let summary = "Call to nd_range::get_global_range";
  let description = [{
    This operation represents a call to the nd_range::get_global_range function.
  }];

  let arguments = (ins Arg<NDRangeMemRef, "The input ND-range", [MemRead]>:$ND);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_range.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNdRangeGetLocalRange
    : SYCLMethodOpInterfaceImpl<"nd_range.get_local_range", "NdRangeType",
                                ["get_local_range"]> {
  let summary = "Call to nd_range::get_local_range";
  let description = [{
    This operation represents a call to the nd_range::get_local_range function.
  }];

  let arguments = (ins Arg<NDRangeMemRef, "The input ND-range", [MemRead]>:$ND);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_range.get_group_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNdRangeGetGroupRange
    : SYCLMethodOpInterfaceImpl<"nd_range.get_group_range", "NdRangeType",
                                ["get_group_range"]> {
  let summary = "Call to nd_range::get_group_range";
  let description = [{
    This operation represents a call to the nd_range::get_group_range function.
  }];

  let arguments = (ins Arg<NDRangeMemRef, "The input ND-range", [MemRead]>:$ND);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// id.constructor OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLIDConstructorOp : SYCLConstructor<"id"> {
  let summary = "Operation returning a new sycl.id value";
  let description = [{
    This operation creates a new `sycl.id`. The following signatures are
    provided:
    ```mlir
    # 0-initialize object:
    () -> memref<1x!sycl_id_N>
    # Initialize each dimension:
    (index) -> memref<1x!sycl_id_1>
    (index, index) -> memref<1x!sycl_id_2>
    (index, index, index) -> memref<1x!sycl_id_3>
    # Copy the dimensions of the input:
    (memref<?x!sycl_id_N>) -> memref<1x!sycl_id_N>
    (memref<?x!sycl_range_N>) -> memref<1x!sycl_id_N>
    # Construct from `sycl.item.get_id`'s result:
    (memref<?x!sycl_item_N>) -> memref<1x!sycl_item_N>
    ```
    In addition of initializing the values, this operation also allocates the
    object.
  }];
  let arguments = (ins Arg<Variadic<AnyType>, "Constructor arguments">:$Args);
  let results =
      (outs Res<MemRefRankOf<[SYCL_IDType], [1]>, "constructed object">:$id);
  let extraClassDeclaration = [{
    /// Return the memory effects associated with each argument.
    ///
    /// Each argument will have `MemoryEffects::Read` if they're a
    /// pointer/memref and no memory effect otherwise.
    void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                        ::mlir::MemoryEffects::Effect>> &effects);
  }];
  let hasVerifier = true;
  let hasFolder = true;
}

////////////////////////////////////////////////////////////////////////////////
// id.get OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLIDGetOp
    : SYCLMethodOpInterfaceImpl<"id.get", "IDType",
                                ["get", "operator[]", "operator unsigned long"],
                                [SYCLGetComponent]> {
  let summary = "Call to id::get/operator[]/operator size_t";
  let description = [{
    This operation represents a call to the id::get/operator[]/operator size_t functions.
  }];

  let arguments = (ins Arg<IDMemRef, "The input ID", [MemRead]>:$ID,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetResult:$Res);

  let assemblyFormat = [{
    $ID `[` $Index `]` attr-dict `:` functional-type(operands, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// item.get_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLItemGetIDOp
    : SYCLMethodOpInterfaceImpl<"item.get_id", "ItemType",
                                ["get_id", "operator[]", "operator unsigned long"],
                                [SYCLGetID, ResultsNonUniform]> {
  let summary = "Call to item::get_id/operator[]/operator size_t";
  let description = [{
    This operation represents a call to the item::get_id/operator[]/operator size_t functions.
  }];

  let arguments = (ins Arg<ItemMemRef, "The input item", [MemRead]>:$Item,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// item.get_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLItemGetRangeOp
    : SYCLMethodOpInterfaceImpl<"item.get_range", "ItemType",
                                ["get_range"], 
                                [SYCLGetRange]> {
  let summary = "Call to item::get_range";
  let description = [{
    This operation represents a call to the item::get_range function.
  }];

  let arguments = (ins Arg<ItemMemRef, "The input item", [MemRead]>:$Item,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// item.get_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLItemGetLinearIDOp
    : SYCLMethodOpInterfaceImpl<"item.get_linear_id", "ItemType",
                                ["get_linear_id"], 
                                [ResultsNonUniform]> {
  let summary = "Call to item::get_linear_id";
  let description = [{
    This operation represents a call to the item::get_linear_id function.
  }];

  let arguments = (ins Arg<ItemMemRef, "The input item", [MemRead]>:$Item);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_global_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGlobalIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_global_id", "NdItemType",
                                ["get_global_id"], 
                                [SYCLGetID, ResultsNonUniform]> {
  let summary = "Call to nd_item::get_global_id";
  let description = [{
    This operation represents a call to the nd_item::get_global_id function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_global_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGlobalLinearIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_global_linear_id", "NdItemType",
                                ["get_global_linear_id"], 
                                [ResultsNonUniform]> {
  let summary = "Call to nd_item::get_global_linear_id";
  let description = [{
    This operation represents a call to the nd_item::get_global_linear_id function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetLocalIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_local_id", "NdItemType",
                                ["get_local_id"], 
                                [SYCLGetID, ResultsNonUniform]> {
  let summary = "Call to nd_item::get_local_id";
  let description = [{
    This operation represents a call to the nd_item::get_local_id function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetLocalLinearIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_local_linear_id", "NdItemType",
                                ["get_local_linear_id"],
                                [ResultsNonUniform]> {
  let summary = "Call to nd_item::get_local_linear_id";
  let description = [{
    This operation represents a call to the nd_item::get_local_linear_id function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_group OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGroupOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_group", "NdItemType",
                                ["get_group"], 
                                [SYCLGetGroup]> {
  let summary = "Call to nd_item::get_group";
  let description = [{
    This operation represents a call to the nd_item::get_group function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs AnyTypeOf<[I64, SYCL_GroupType]>:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_group_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGroupLinearIDOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_group_linear_id", "NdItemType",
                                ["get_group_linear_id"],
                                [ResultsNonUniform]> {
  let summary = "Call to nd_item::get_group_linear_id";
  let description = [{
    This operation represents a call to the nd_item::get_group_linear_id function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_group_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGroupRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_group_range", "NdItemType",
                                ["get_group_range"],
                                [SYCLGetRange]> {
  let summary = "Call to nd_item::get_group_range";
  let description = [{
    This operation represents a call to the nd_item::get_group_range function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_global_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetGlobalRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_global_range", "NdItemType",
                                ["get_global_range"],
                                [SYCLGetRange]> {
  let summary = "Call to nd_item::get_global_range";
  let description = [{
    This operation represents a call to the nd_item::get_global_range function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetLocalRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_local_range", "NdItemType",
                                ["get_local_range"],
                                [SYCLGetRange]> {
  let summary = "Call to nd_item::get_local_range";
  let description = [{
    This operation represents a call to the nd_item::get_local_range function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// nd_item.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLNDItemGetNdRangeOp
    : SYCLMethodOpInterfaceImpl<"nd_item.get_nd_range", "NdItemType",
                                ["get_nd_range"]> {
  let summary = "Call to nd_item::get_nd_range";
  let description = [{
    This operation represents a call to the nd_item::get_nd_range function.
  }];

  let arguments = (ins Arg<NDItemMemRef, "The input ND-item", [MemRead]>:$NDItem);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCL_NdRangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetGroupIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_id", "GroupType",
                                ["get_group_id", "operator[]"],
                                [SYCLGetID, ResultsNonUniform]> {
  let summary = "Call to group::get_group_id/operator[]";
  let description = [{
    This operation represents a call to the group::get_group_id/operator[] functions.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_id", "GroupType",
                                ["get_local_id"],
                                [SYCLGetID,  ResultsNonUniform]> {
  let summary = "Call to group::get_local_id";
  let description = [{
    This operation represents a call to the group::get_local_id function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetIDResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_range", "GroupType",
                                ["get_local_range"],
                                [SYCLGetRange]> {
  let summary = "Call to group::get_local_range";
  let description = [{
    This operation represents a call to the group::get_local_range function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetGroupRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_range", "GroupType",
                                ["get_group_range"],
                                [SYCLGetRange]> {
  let summary = "Call to group::get_group_range";
  let description = [{
    This operation represents a call to the group::get_group_range function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group,
                       Optional<I32>:$Index);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCLGetRangeResult:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_max_local_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetMaxLocalRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_max_local_range", "GroupType",
                                ["get_max_local_range"]> {
  let summary = "Call to group::get_max_local_range";
  let description = [{
    This operation represents a call to the group::get_max_local_range function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs SYCL_RangeType:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetGroupLinearIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_linear_id", "GroupType",
                                ["get_group_linear_id"],
                                [ResultsNonUniform]> {
  let summary = "Call to group::get_group_linear_id";
  let description = [{
    This operation represents a call to the group::get_group_linear_id function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_linear_id OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalLinearIDOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_linear_id", "GroupType",
                                ["get_local_linear_id"],
                                [ResultsNonUniform]> {
  let summary = "Call to group::get_local_linear_id";
  let description = [{
    This operation represents a call to the group::get_local_linear_id function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_group_linear_range OPERATION
////////////////////////////////////////////////////////////////////////////////

 def SYCLGroupGetGroupLinearRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_group_linear_range", "GroupType",
                                ["get_group_linear_range"]> {
  let summary = "Call to group::get_group_linear_range";
  let description = [{
    This operation represents a call to the group::get_group_linear_range function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

////////////////////////////////////////////////////////////////////////////////
// group.get_local_linear_range OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLGroupGetLocalLinearRangeOp
    : SYCLMethodOpInterfaceImpl<"group.get_local_linear_range", "GroupType",
                                ["get_local_linear_range"]> {
  let summary = "Call to group::get_local_linear_range";
  let description = [{
    This operation represents a call to the group::get_local_linear_range function.
  }];

  let arguments = (ins Arg<GroupMemRef, "The input group", [MemRead]>:$Group);

  let extraClassDeclaration = extraClassDeclarationBase;

  let results = (outs I64:$Res);
}

include "mlir/Dialect/SYCL/IR/SYCLAuxOps.td"
include "mlir/Dialect/SYCL/IR/SYCLHostOps.td"
include "mlir/Dialect/SYCL/IR/SYCLMathOps.td"

#endif // SYCL_OPS
