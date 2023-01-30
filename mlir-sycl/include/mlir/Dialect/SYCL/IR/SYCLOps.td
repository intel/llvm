//===--- SYCLOps.td -------------------------------------------------------===//
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
include "mlir/IR/AttrTypeBase.td"

include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

class SYCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<SYCL_Dialect, mnemonic, traits>;

class SYCLMethodOpInterfaceImpl<
    string mnemonic, string type, list<string> methodNames, list<Trait> traits = []>
        : SYCL_Op<mnemonic, !listconcat(traits, [SYCLMethodOpInterface, CallOpInterface])> {
  string baseType = type;
  list<string> memberFunctionNames = methodNames;
  int arrSize = !size(memberFunctionNames);

  let extraClassDeclaration = [{
    static constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> methods{{ }] #
      !interleave(!foreach(name, memberFunctionNames, "\"" # name # "\""), ", ") # [{
     }};
    static ::mlir::TypeID getTypeID() { return ::mlir::sycl::}] # type # [{::getTypeID(); }
    static constexpr llvm::ArrayRef<llvm::StringLiteral> getMethodNames() { return methods; }

    Operation::operand_iterator arg_operand_begin() { return (*this)->operand_begin(); }
    Operation::operand_iterator arg_operand_end() { return (*this)->operand_end(); }

    /// Return the callee of the generic SYCL call operation, this is required by
    /// the call interface.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<FlatSymbolRefAttr>(getMangledFunctionNameAttrName());
    }

    /// Get the argument operands to the called function, this is required by the
    /// call interface.
    Operation::operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
  }];

  let extraClassDefinition = [{
    constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> $cppClass::methods;
  }];
}

////////////////////////////////////////////////////////////////////////////////
// TYPES DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

class SYCL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SYCL_Dialect, name,
              !listconcat([MemRefElementTypeInterface,
                           LLVM_PointerElementTypeInterface],
                               traits)> {
  let mnemonic = typeMnemonic;
}

class SYCLInheritanceTypeTrait<string ParentClass>
    : NativeTypeTrait<"SYCLInheritanceTypeTrait<" # ParentClass # ">::Trait"> {
  let cppNamespace = "::mlir::sycl";
}

//
// SYCL types (alphabetical order).
//
def SYCL_AccessorCommonType : SYCL_Type<"AccessorCommon", "accessor_common"> {
  let assemblyFormat = "";
}

def SYCL_AccessorImplDeviceType :
    SYCL_Type<"AccessorImplDevice", "accessor_impl_device"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_AccessorType
    : SYCL_Type<"Accessor", "accessor",
        [SYCLInheritanceTypeTrait<"AccessorCommonType">,
         SYCLInheritanceTypeTrait<"LocalAccessorBaseType">,
         SYCLInheritanceTypeTrait<"OwnerLessBaseType">]> {
  let summary = "Buffer Accessor";
  let description = [{ Accessor to a buffer object.}];
  let parameters = (ins "::mlir::Type":$type,
                        "unsigned":$dimension,
                        "mlir::sycl::MemoryAccessMode":$accessMode,
                        "mlir::sycl::MemoryTargetMode":$targetMode,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat =
      "`<` `[` $dimension `,` $type `,` ` ` custom<MemoryAccessMode>($accessMode) `,` ` ` custom<MemoryTargetMode>($targetMode) `]` `,` `(` $body `)` `>`";
}

def SYCL_AccessorSubscriptType :
    SYCL_Type<"AccessorSubscript", "accessor_subscript"> {
  let parameters = (ins "int":$currentDimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $currentDimension `]` `,` `(` $body `)` `>`";
}

def SYCL_ArrayType : SYCL_Type<"Array", "array"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_AssertHappenedType : SYCL_Type<"AssertHappened", "assert_happened"> {
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `(` $body `)` `>`";
}

def SYCL_AtomicType : SYCL_Type<"Atomic", "atomic"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        "mlir::sycl::AccessAddrSpace":$addrSpace,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` custom<AccessAddrSpace>($addrSpace) `]` `,` `(` $body `)` `>`";
}

def SYCL_BFloat16Type : SYCL_Type<"BFloat16", "bfloat16"> {
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `(` $body `)` `>`";
}

def SYCL_GetOpType : SYCL_Type<"GetOp", "get_op"> {
  let parameters = (ins "::mlir::Type":$dataType);
  let assemblyFormat = "`<` $dataType `>`";
}

def SYCL_GetScalarOpType : SYCL_Type<"GetScalarOp", "get_scalar_op"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `]` `,` `(` $body `)` `>`";
}

def SYCL_GroupType : SYCL_Type<"Group", "group"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_HItemType : SYCL_Type<"HItem", "h_item"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_IDType
    : SYCL_Type<"ID", "id", [SYCLInheritanceTypeTrait<"ArrayType">]> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<`  `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_ItemBaseType : SYCL_Type<"ItemBase", "item_base"> {
  let parameters = (ins "unsigned":$dimension,
                        "bool":$withOffset,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `,` $withOffset `]` `,` `(` $body `)` `>`";
}

def SYCL_ItemType : SYCL_Type<"Item", "item"> {
  let parameters = (ins "unsigned":$dimension,
                        "bool":$withOffset,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `,` $withOffset `]` `,` `(` $body `)` `>`";
}

def SYCL_KernelHandlerType : SYCL_Type<"KernelHandler", "kernel_handler"> {
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `(` $body `)` `>`";
}

def SYCL_LocalAccessorBaseDeviceType
    : SYCL_Type<"LocalAccessorBaseDevice", "LocalAccessorBaseDevice"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";                        
}

def SYCL_LocalAccessorBaseType
    : SYCL_Type<"LocalAccessorBase", "local_accessor_base",
        [SYCLInheritanceTypeTrait<"AccessorCommonType">]> {
  let parameters = (ins "::mlir::Type":$type,
                        "unsigned":$dimension,
                        "mlir::sycl::MemoryAccessMode":$accessMode,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat =
      "`<` `[` $dimension `,` $type `,` ` ` custom<MemoryAccessMode>($accessMode) `]` `,` `(` $body `)` `>`";
}

def SYCL_LocalAccessorType
    : SYCL_Type<"LocalAccessor", "local_accessor",
        [SYCLInheritanceTypeTrait<"LocalAccessorBaseType">]> {
  let summary = "Local Accessor";
  let description = [{ Accessor to a local buffer.}];
  let parameters = (ins "::mlir::Type":$type,
                        "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat =
      "`<` `[` $dimension `,` $type `]` `,` `(` $body `)` `>`";
}

def SYCL_MaximumType : SYCL_Type<"Maximum", "maximum"> {
  let parameters = (ins "::mlir::Type":$dataType);
  let assemblyFormat = "`<` $dataType `>`";
}

def SYCL_MinimumType : SYCL_Type<"Minimum", "minimum"> {
  let parameters = (ins "::mlir::Type":$dataType);
  let assemblyFormat = "`<` $dataType `>`";
}

def SYCL_MultiPtrType : SYCL_Type<"MultiPtr", "multi_ptr"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        "mlir::sycl::AccessAddrSpace":$addrSpace,
                        "mlir::sycl::DecoratedAccess":$decAccess,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` ` ` custom<AccessAddrSpace>($addrSpace) `,` ` ` custom<DecoratedAccess>($decAccess) `]` `,` `(` $body `)` `>`";
}

def SYCL_NdItemType : SYCL_Type<"NdItem", "nd_item"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_NdRangeType : SYCL_Type<"NdRange", "nd_range"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_OwnerLessBaseType : SYCL_Type<"OwnerLessBase", "owner_less_base"> {
  let assemblyFormat = "";
}

def SYCL_RangeType
    : SYCL_Type<"Range", "range", [SYCLInheritanceTypeTrait<"ArrayType">]> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<`  `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_StreamType : SYCL_Type<"Stream", "stream"> {
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `(` $body `)` `>`";
}

def SYCL_SubGroupType : SYCL_Type<"SubGroup", "sub_group"> {
  let assemblyFormat = "";
}

def SYCL_SwizzledVecType : SYCL_Type<"SwizzledVec", "swizzled_vec"> {
  let parameters = (ins "::mlir::sycl::VecType":$vecType,
                        ArrayRefParameter<"int">:$indexes,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $vecType `,` $indexes `]` `,` `(` $body `)` `>`";
}

def SYCL_TupleCopyAssignableValueHolderType
    : SYCL_Type<"TupleCopyAssignableValueHolder", "tuple_copy_assignable_value_holder",
        [SYCLInheritanceTypeTrait<"TupleValueHolderType">]> {
  let parameters = (ins "::mlir::Type":$dataType,
                        "bool":$isTriviallyCopyAssignable,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` $isTriviallyCopyAssignable `]` `,` `(` $body `)` `>`";
}

def SYCL_TupleValueHolderType : SYCL_Type<"TupleValueHolder", "tuple_value_holder"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `]` `,` `(` $body `)` `>`";
}

def SYCL_VecType : SYCL_Type<"Vec", "vec"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        "int":$numElements,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` $numElements `]` `,` `(` $body `)` `>`";
  let genVerifyDecl = 1;
}

// Define MemRef types (alphabetical order).
def AccessorCommonMemRef : MemRefOf<[SYCL_AccessorCommonType]>;
def AccessorImplDeviceMemRef : MemRefOf<[SYCL_AccessorImplDeviceType]>;
def AccessorMemRef : MemRefOf<[SYCL_AccessorType]>;
def AccessorSubscriptMemRef : MemRefOf<[SYCL_AccessorSubscriptType]>;
def ArrayMemRef : MemRefOf<[SYCL_ArrayType]>;
def AtomicMemRef : MemRefOf<[SYCL_AtomicType]>;
def GetOpMemRef : MemRefOf<[SYCL_GetOpType]>;
def GetScalarOpMemRef : MemRefOf<[SYCL_GetScalarOpType]>;
def GroupMemRef : MemRefOf<[SYCL_GroupType]>;
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
  GetOpMemRef,
  GetScalarOpMemRef,
  GroupMemRef,  
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
def SYCLIndexSpaceGetIDResult : AnyTypeOf<[Index, SYCL_IDType]>;
def SYCLIndexSpaceGetRangeResult : AnyTypeOf<[Index, SYCL_RangeType]>;
def SYCLGetResult : AnyTypeOf<[I64, MemRefOf<[I64]>]>;
def VectorEltTy : AnyTypeOf<[I1, I8, I16, I32, I64, F16, F32, F64]>;
def VectorSplatArg : MemRefOf<[VectorEltTy]>;

////////////////////////////////////////////////////////////////////////////////
// SYCL GRID OPERATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCLNumWorkItemsOp : SYCL_Op<"num_work_items", [SYCLIndexSpaceGetRange]> {
  let summary = "Retrieve the number of work-items.";
  let description = [{
    This operation returns the number of work-items in the index space. If the
    optional argument is passed, the number of work-items in the given dimension
    is returned.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetRangeResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLGlobalIDOp : SYCL_Op<"global_id", [SYCLIndexSpaceGetID]> {
  let summary = "Retrieve the global ID of the item.";
  let description = [{
    This operation returns the global ID of the item in the index space. If the
    optional argument is passed, the global ID of the item in the given
    dimension is returned.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetIDResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLLocalIDOp : SYCL_Op<"local_id", [SYCLIndexSpaceGetID]> {
  let summary = "Retrieve the local ID of the item.";
  let description = [{
    This operation returns the local ID of the item within the work-group. If
    the optional argument is passed, the local ID of the item in the given
    dimension is returned.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetIDResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLGlobalOffsetOp : SYCL_Op<"global_offset", [SYCLIndexSpaceGetID]>,
                       Deprecated<"deprecated in SYCL 2020"> {
  let summary = "Retrieve the global offset of the item.";
  let description = [{
    This operation returns the global offset of the item in the index space. If
    the optional argument is passed, the global offset of the item in the given
    dimension is returned.

    Note that this is deprecated in SYCL 2020.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetIDResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLNumWorkGroupsOp : SYCL_Op<"num_work_groups", [SYCLIndexSpaceGetRange]> {
  let summary = "Retrieve the number of work-groups.";
  let description = [{
    This operation returns the number of work-groups in the index space. If the
    optional argument is passed, the number of work-groups in the given
    dimension is returned.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetRangeResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLWorkGroupSizeOp : SYCL_Op<"work_group_size", [SYCLIndexSpaceGetRange]> {
  let summary = "Retrieve the number of work-items in a work-group.";
  let description = [{
    This operation returns the number of work-items per work-group. If the
    optional argument is passed, the number of work-items per work-group in the
    given dimension is returned.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetRangeResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLWorkGroupIDOp : SYCL_Op<"work_group_id", [SYCLIndexSpaceGetID]> {
  let summary = "Retrieve the ID of the work-group.";
  let description = [{
    This operation returns the ID of the work-group. If the optional argument is
    passed, the ID of the work-group in the given dimension is returned.
  }];

  let arguments = (ins Optional<I32>:$dimension);
  let results = (outs SYCLIndexSpaceGetIDResult:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLNumSubGroupsOp : SYCL_Op<"num_sub_groups"> {
  let summary = "Retrieve the number of sub-groups.";
  let description = [{
    This operation returns the number of sub-groups.
  }];

  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLSubGroupMaxSizeOp : SYCL_Op<"sub_group_max_size"> {
  let summary = "Retrieve the maximum size of a sub-group.";
  let description = [{
    This operation returns the maximum size of a sub-group.
  }];

  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLSubGroupSizeOp : SYCL_Op<"sub_group_size"> {
  let summary = "Retrieve the sub-group size.";
  let description = [{
    This operation returns the sub-group size.
  }];

  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLSubGroupIDOp : SYCL_Op<"sub_group_id"> {
  let summary = "Retrieve the ID of the sub-group.";
  let description = [{
    This operation returns the ID of the sub-group.
  }];

  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    attr-dict `:` functional-type(operands, results)
  }];
}

def SYCLSubGroupLocalIDOp : SYCL_Op<"sub_group_local_id"> {
  let summary = "Retrieve the local ID of the sub-group.";
  let description = [{
    This operation returns the local ID of the sub-group.
  }];

  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    attr-dict `:` functional-type(operands, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR OPERATION
////////////////////////////////////////////////////////////////////////////////

def ConstructorArgs : AnyTypeOf<[SYCLMemref,
                                 VectorEltTy,
                                 IndexType,
                                 VectorSplatArg,
                                 SYCL_IDType,
                                 SYCL_RangeType,
                                 Builtin_Vector,
                               ]>;
def SYCLConstructorOp : SYCL_Op<"constructor", [CallOpInterface]> {
  let summary = "Generic constructor operation";
  let description = [{
    This operation represent the call to the constructor of a SYCL type.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$TypeName,
    FlatSymbolRefAttr:$MangledFunctionName,
    Variadic<ConstructorArgs>:$Args
  );
  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of the generic SYCL call operation, this is required by
    /// the call interface.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<FlatSymbolRefAttr>(getMangledFunctionNameAttrName());
    }

    /// Get the argument operands to the called function, this is required by the
    /// call interface.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
  }];      

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// CAST OPERATION
////////////////////////////////////////////////////////////////////////////////

def CastArg : AnyTypeOf<[IDMemRef, RangeMemRef, AccessorMemRef, LocalAccessorMemRef, LocalAccessorBaseMemRef]>;
def CastRes : AnyTypeOf<[ArrayMemRef, AccessorCommonMemRef, LocalAccessorBaseMemRef, OwnerLessBaseMemRef]>;
def SYCLCastOp : SYCL_Op<"cast", [DeclareOpInterfaceMethods<CastOpInterface>, 
  Pure, MemRefsNormalizable]
> {
  let summary = "Derived type to base type cast operation";
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

    /// Get the argument operands to the called function, this is required by the
    /// call interface.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
  }];    

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

////////////////////////////////////////////////////////////////////////////////
// accessor.subscript OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLAccessorSubscriptIndex : AnyTypeOf<[IndexType, SYCL_IDType]>;

def SYCLAccessorSubscriptOp
    : SYCLMethodOpInterfaceImpl<"accessor.subscript", "AccessorType",
                                ["operator[]"], [Pure]> {
  let summary = "Call to accessor::operator[]";
  let description = [{
    This operation represents a call to the accessor::operator[] function.
  }];

  let arguments = (ins SYCL_AccessorType:$Acc,
                       SYCLAccessorSubscriptIndex:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_RangeType:$Range,
                       I32:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_RangeType:$Range,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdRangeType:$ND,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdRangeType:$ND,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdRangeType:$ND,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_IDType:$ID,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_ItemType:$Item,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_ItemType:$Item,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_ItemType:$Item,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_NdItemType:$NDItem,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       Optional<I32>:$Index,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
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

  let arguments = (ins SYCL_GroupType:$Group,
                       TypeArrayAttr:$ArgumentTypes,
                       FlatSymbolRefAttr:$FunctionName,
                       OptionalAttr<FlatSymbolRefAttr>:$MangledFunctionName,
                       FlatSymbolRefAttr:$TypeName);

  let results = (outs I64:$Res);
}

#endif // SYCL_OPS
