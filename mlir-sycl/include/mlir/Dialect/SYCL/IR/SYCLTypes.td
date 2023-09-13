//===-- SYCLTypes.td - SYCL dialect type definitions *- tablegen -*--------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for SYCL dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_TYPES
#define SYCL_TYPES

include "mlir/IR/AttrTypeBase.td"

include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Dialect/SYCL/IR/SYCLBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

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
                        "::mlir::sycl::AccessMode":$accessMode,
                        "::mlir::sycl::Target":$targetMode,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat =
      "`<` `[` $dimension `,` $type `,` ` ` $accessMode `,` ` ` $targetMode `]` `,` `(` $body `)` `>`";
}

def SYCL_BufferType : SYCL_Type<"Buffer", "buffer"> {
  let summary = "SYCL Buffer";
  let description = [{ Shared array of one, two or three dimensions,
                        accessed through accessors.}];
  let parameters = (ins "::mlir::Type":$type,
                        "unsigned":$dimension,
                        DefaultValuedParameter<"bool", "false">:$subbuffer);
  let assemblyFormat = 
      "`<` `[` $dimension `,` $type (`,` $subbuffer^)? `]` `>`";
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

def SYCL_AtomicType : SYCL_Type<"Atomic", "atomic"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        "::mlir::sycl::AccessAddrSpace":$addrSpace,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` $addrSpace `]` `,` `(` $body `)` `>`";
}

def SYCL_GroupType : SYCL_Type<"Group", "group"> {
  let parameters = (ins "unsigned":$dimension,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dimension `]` `,` `(` $body `)` `>`";
}

def SYCL_HalfType : SYCL_Type<"Half", "half"> {
  let summary = "sycl::half";
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `(` $body `)` `>`";
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
                        "::mlir::sycl::AccessMode":$accessMode,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat =
      "`<` `[` $dimension `,` $type `,` ` ` $accessMode `]` `,` `(` $body `)` `>`";
}

def SYCL_LocalAccessorType
    : SYCL_Type<"LocalAccessor", "local_accessor",
        [SYCLInheritanceTypeTrait<"LocalAccessorBaseType">,
         SYCLInheritanceTypeTrait<"OwnerLessBaseType">]> {
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
                        "::mlir::sycl::AccessAddrSpace":$addrSpace,
                        "::mlir::sycl::AccessDecorated":$decAccess,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` ` ` $addrSpace `,` ` ` $decAccess `]` `,` `(` $body `)` `>`";
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
def SYCL_VecType : SYCL_Type<"Vec", "vec"> {
  let parameters = (ins "::mlir::Type":$dataType,
                        "int":$numElements,
                        ArrayRefParameter<"mlir::Type">:$body);
  let assemblyFormat = "`<` `[` $dataType `,` $numElements `]` `,` `(` $body `)` `>`";
  let genVerifyDecl = 1;
}

#endif // SYCL_TYPES
