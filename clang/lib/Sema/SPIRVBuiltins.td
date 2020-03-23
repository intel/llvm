//==--- SPIRVBuiltins.td - SPIRV builtin declarations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains TableGen definitions for SPIR-V builtin function
// declarations.  In case of an unresolved function name, Clang will check for
// a function described in this file when -fdeclare-spirv-builtins is specified.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//              Definitions of miscellaneous basic entities.
//===----------------------------------------------------------------------===//
// TODO: basic entities declaration with OpenCLBuiltins.td

// TODO: Manage version using the JSON grammar.  Unused for now.
class Version<int _Version> {
  int ID = _Version;
}
def SPIRVAll : Version<  0>;

// Address spaces
// Pointer types need to be assigned an address space.
class AddressSpace<string _AS> {
  string Name = _AS;
}
def DefaultAS    : AddressSpace<"clang::LangAS::Default">;
def PrivateAS    : AddressSpace<"clang::LangAS::sycl_private">;
def GlobalAS     : AddressSpace<"clang::LangAS::sycl_global">;
def ConstantAS   : AddressSpace<"clang::LangAS::sycl_constant">;
def LocalAS      : AddressSpace<"clang::LangAS::sycl_local">;
def GenericAS    : AddressSpace<"clang::LangAS::sycl_generic">;

// TODO: Manage capabilities. Unused for now.
class AbstractExtension<string _Ext> {
  string ExtName = _Ext;
}

// Extension associated to a builtin function.
class FunctionExtension<string _Ext> : AbstractExtension<_Ext>;

// FunctionExtension definitions.
def FuncExtNone                          : FunctionExtension<"">;

// Qualified Type.  These map to ASTContext::QualType.
class QualType<string _Name, bit _IsAbstract=0, bit _IsSigned=0> {
  // Name of the field or function in a clang::ASTContext
  // E.g. Name="IntTy" for the int type, and "getIntPtrType()" for an intptr_t
  string Name = _Name;
  // Some QualTypes in this file represent an abstract type for which there is
  // no corresponding AST QualType, e.g. a GenType or an `image2d_t` type
  // without access qualifiers.
  bit IsAbstract = _IsAbstract;
  bit IsSigned = _IsSigned;
}

// List of integers.
class IntList<string _Name, list<int> _List> {
  string Name = _Name;
  list<int> List = _List;
}

// Basic data types (int, float, image2d_t, ...).
// Its child classes can represent concrete types (e.g. VectorType) or
// abstract types (e.g. GenType).
class Type<string _Name, QualType _QTName> {
  // Name of the Type.
  string Name = _Name;
  // QualType associated with this type.
  QualType QTName = _QTName;
  // Size of the vector (if applicable).
  int VecWidth = 1;
  // Size of the element in bits.
  int ElementSize = 1;
  // Is a integer.
  bit IsInteger = 0;
  // Is a signed integer.
  bit IsSigned = 1;
  // Is a float.
  bit IsFloat = 0;
  // Is a pointer.
  bit IsPointer = 0;
  // "const" qualifier.
  bit IsConst = 0;
  // "volatile" qualifier.
  bit IsVolatile = 0;
  // Access qualifier. Must be one of ("RO", "WO", "RW").
  string AccessQualifier = "";
  // Address space.
  string AddrSpace = DefaultAS.Name;
}

class FundamentalType<string _Name, QualType _QTName, int _Size> : Type<_Name, _QTName> {
  // Inherited fields
  let ElementSize = _Size;
}

// Integer Type.
class IntType<string _Name, QualType _QTName, int _Size> : FundamentalType<_Name, _QTName, _Size> {
  // Inherited fields
  let IsInteger = 1;
  let IsSigned = 1;
}

// Unsigned integer Type.
class UIntType<string _Name, QualType _QTName, int _Size> : FundamentalType<_Name, _QTName, _Size> {
  // Inherited fields
  let IsInteger = 1;
  let IsSigned = 0;
}

// Floating Type.
class FPType<string _Name, QualType _QTName, int _Size> : FundamentalType<_Name, _QTName, _Size> {
  // Inherited fields
  let IsFloat = 1;
}

class CompoundType<Type _Ty> : Type<_Ty.Name, _Ty.QTName> {
  // Inherited fields
  let VecWidth = _Ty.VecWidth;
  let ElementSize = _Ty.ElementSize;
  let IsInteger = _Ty.IsInteger;
  let IsSigned = _Ty.IsSigned;
  let IsFloat = _Ty.IsFloat;
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AccessQualifier = _Ty.AccessQualifier;
  let AddrSpace = _Ty.AddrSpace;

  Type ElementType = _Ty;
}

// Vector types (e.g. int2, int3, int16, float8, ...).
class VectorType<Type _Ty, int _VecWidth> : Type<_Ty.Name, _Ty.QTName> {
  let VecWidth = _VecWidth;
  let AccessQualifier = "";
  // Inherited fields
  let ElementSize = _Ty.ElementSize;
  let IsInteger = _Ty.IsInteger;
  let IsSigned = _Ty.IsSigned;
  let IsFloat = _Ty.IsFloat;
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AccessQualifier = _Ty.AccessQualifier;
  let AddrSpace = _Ty.AddrSpace;
}

// Pointer types (e.g. int*, float*, ...).
class PointerType<Type _Ty, AddressSpace _AS = DefaultAS> :
    CompoundType<_Ty> {
  // Inherited fields
  let IsPointer = 1;
  let AddrSpace = _AS.Name;
}

// Const types (e.g. const int).
class ConstType<Type _Ty> : CompoundType<_Ty> {
  // Inherited fields
  let IsConst = 1;
}

// Volatile types (e.g. volatile int).
class VolatileType<Type _Ty> : CompoundType<_Ty> {
  // Inherited fields
  let IsVolatile = 1;
}

// Image types (e.g. image2d).
class ImageType<Type _Ty, string _AccessQualifier> :
    Type<_Ty.Name, QualType<_Ty.QTName.Name#_AccessQualifier#"Ty", 0>> {
  let VecWidth = 0;
  let AccessQualifier = _AccessQualifier;
  // Inherited fields
  let ElementSize = _Ty.ElementSize;
  let IsInteger = _Ty.IsInteger;
  let IsSigned = _Ty.IsSigned;
  let IsFloat = _Ty.IsFloat;
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AddrSpace = _Ty.AddrSpace;
}

// List of Types.
class TypeList<list<Type> _Type> {
  list<Type> List = _Type;
}

// A GenericType is an abstract type that defines a set of types as a
// combination of Types and vector sizes.
//
// For example, if TypeList = <int, float> and VectorList = <1, 2, 4>, then it
// represents <int, int2, int4, float, float2, float4>.
//
// Some rules apply when using multiple GenericType arguments in a declaration:
//   1. The number of vector sizes must be equal or 1 for all gentypes in a
//      declaration.
//   2. The number of Types must be equal or 1 for all gentypes in a
//      declaration.
//   3. Generic types are combined by iterating over all generic types at once.
//      For example, for the following GenericTypes
//        GenT1 = GenericType<half, [1, 2]> and
//        GenT2 = GenericType<float, int, [1, 2]>
//      A declaration f(GenT1, GenT2) results in the combinations
//        f(half, float), f(half2, float2), f(half, int), f(half2, int2) .
//   4. "sgentype" from the OpenCL specification is supported by specifying
//      a single vector size.
//      For example, for the following GenericTypes
//        GenT = GenericType<half, int, [1, 2]> and
//        SGenT = GenericType<half, int, [1]>
//      A declaration f(GenT, SGenT) results in the combinations
//        f(half, half), f(half2, half), f(int, int), f(int2, int) .
class GenericType<string _Ty, TypeList _TypeList, IntList _VectorList> :
    Type<_Ty, QualType<"null", 1>> {
  // Possible element types of the generic type.
  TypeList TypeList = _TypeList;
  // Possible vector sizes of the types in the TypeList.
  IntList VectorList = _VectorList;
  // The VecWidth field is ignored for GenericTypes. Use VectorList instead.
  let VecWidth = 0;
}

// Builtin function attributes.
def Attr {
  list<bit> None = [0, 0, 0];
  list<bit> Pure = [1, 0, 0];
  list<bit> Const = [0, 1, 0];
  list<bit> Convergent = [0, 0, 1];
}

//===----------------------------------------------------------------------===//
//                      Class for builtin functions
//===----------------------------------------------------------------------===//
class Builtin<string _Name, list<Type> _Signature, list<bit> _Attributes = Attr.None> {
  // Name of the builtin function
  string Name = _Name;
  // List of types used by the function. The first one is the return type and
  // the following are the arguments. The list must have at least one element
  // (the return type).
  list<Type> Signature = _Signature;
  // Function attribute __attribute__((pure))
  bit IsPure = _Attributes[0];
  // Function attribute __attribute__((const))
  bit IsConst = _Attributes[1];
  // Function attribute __attribute__((convergent))
  bit IsConv = _Attributes[2];
  // Is function a variadic one
  bit IsVariadic = 0;
  // OpenCL extensions to which the function belongs.
  FunctionExtension Extension = FuncExtNone;
  // Version from which the function is available.
  // MinVersion is inclusive.
  Version MinVersion = SPIRVAll;
  // Version from which the function is not supported anymore.
  // MaxVersion is exclusive.
  // SPIRVAll makes the function available for all versions.
  Version MaxVersion = SPIRVAll;
}

// Helper to declare SPIR-V Core builtins.
class SPVBuiltin<string _Name, list<Type> _Signature, list<bit> _Attributes = Attr.None> :
Builtin<"__spirv_" # _Name, _Signature, _Attributes> {} 

// Helper to declare OpenCL SPIR-V extended set builtins.
class OCLSPVBuiltin<string _Name, list<Type> _Signature, list<bit> _Attributes = Attr.None> :
SPVBuiltin<"ocl_" # _Name, _Signature, _Attributes> {} 

//===----------------------------------------------------------------------===//
//                 Definitions of types
//===----------------------------------------------------------------------===//

def Float     : FPType<"float",     QualType<"FloatTy">, 32>;
def Double    : FPType<"double",    QualType<"DoubleTy">, 64>;
def Half      : FPType<"half",      QualType<"Float16Ty">, 16>;

//===----------------------------------------------------------------------===//
//                 Definitions of gentype variants
//===----------------------------------------------------------------------===//

// Vector width lists.
def VecAndScalar: IntList<"VecAndScalar", [1, 2, 3, 4, 8, 16]>;

// Type lists.
def TLFloat         : TypeList<[Float, Double, Half]>;

// Float
def FGenTypeN              : GenericType<"FGenTypeN", TLFloat, VecAndScalar>;



//===----------------------------------------------------------------------===//
//                 Definitions of builtins
//           extinst.opencl.std.100.grammar.json
//===----------------------------------------------------------------------===//

// 2.1. Math extended instructions

def : OCLSPVBuiltin<"acos", [FGenTypeN, FGenTypeN], Attr.Const>;

