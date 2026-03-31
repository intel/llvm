//===- SYCLRemangleLibspirv.cpp - Remangle libspirv builtins for SYCL -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that remangles __spirv_* builtin functions to
// provide multiple SYCL mangled variants for different type representations.
// It ensures consistent mangling between libspirv and the target.
//
// Three-Map Transformation:
// For each function the pass applies three type maps in sequence:
//
// - Rename (ParameterTypeReplacements):
//     Rename the function. char -> signed char (always), half -> _Float16,
//     long -> long long, unsigned long -> unsigned long long.
//
// - CloneOriginal (CloneTypeReplacements):
//     After Rename, the original OpenCL-mangled name no longer exists.
//     CloneOriginal recreates it so callers using OpenCL type names still link.
//     Clone source is found by applying CloneOriginal's map to original name:
//       char -> signed char (signed) / unsigned char (unsigned)
//       _Float16 -> half, long -> int (L32) or long long (L64),
//       unsigned long -> unsigned int (L32) or unsigned long long (L64).
//
// - CloneRemangled (RemangledCloneTypeReplacements):
//     When Rename renames a function with multiple parameters, it may absorb
//     valid permutations of the remangled name. For example,
//     `f(long long, char, signed char)` (`_Z1fxca`) is absorbed by
//     `_Z1fxaa` after Rename renames `f(long, char, signed char)`.
//     CloneRemangled creates these missing permutations by cloning the
//     Rename-renamed function. It covers entries where Rename and
//     CloneOriginal differ:
//         char -> signed char, half -> _Float16,
//         long -> long long (L32 only),
//         unsigned long -> unsigned long long (L32 only).
//
// Remangled Pointer Address Space Example:
//   If libspirv defines a function `f(int *)`, the mangled name is
//   `_Z1fPU3AS4i` for a target when generic address space is 4. It is renamed
//   to `_Z1fPi`, to be consistent with SYCL mangling. If libspirv defines a
//   function `f(private int *)`, the mangled name is `_Z1fPi` when default
//   address space is private. The pass renames it to `_Z1fPU3AS0i`.
//
// Type Remappings:
//
// 1. long <-> long long (64-bit platforms)
//    OpenCL C defines only "long" (always 64-bit). SYCL has both "long" and
//    "long long" (both 64-bit on 64-bit platforms). The pass creates both
//    variants to support calls using either type.
//
//    Rename:
//        _Z17__spirv_ocl_s_absl -> _Z17__spirv_ocl_s_absx
//        (long)                 -> (long long)
//
//    CloneTypeReplacements (clone, L64):
//        _Z17__spirv_ocl_s_absx -> _Z17__spirv_ocl_s_absl
//        (long long)            -> (long)
//
// 2. char <-> signed char (when char is signed on host)
//    OpenCL C uses "char" for signed 8-bit. SYCL distinguishes "char" and
//    "signed char". When host char is signed (e.g., x86), both variants are
//    created with the same IR type but different mangling.
//
//    Rename:
//        _Z17__spirv_ocl_s_absc -> _Z17__spirv_ocl_s_absa
//        (char)                 -> (signed char)
//
//    CloneTypeReplacements (clone, signed):
//        _Z17__spirv_ocl_s_absa -> _Z17__spirv_ocl_s_absc
//        (signed char)          -> (char)
//
// 2a. char <-> unsigned char (when char is unsigned on host)
//     When host char is unsigned (e.g., ARM, PowerPC, RISCV), the pass maps
//     'char' to 'unsigned char' to match the host's char signedness.
//
//     Rename:
//         _Z15__spirv_ocl_clzc -> _Z15__spirv_ocl_clza
//         (char)               -> (signed char)
//
//     CloneTypeReplacements (clone, unsigned):
//         _Z15__spirv_ocl_clzh -> _Z15__spirv_ocl_clzc
//         (unsigned char)      -> (char)
//
// 3. half -> _Float16
//    SYCL uses _Float16 vendor extension. OpenCL C uses the 'half' type.
//    The pass renames 'half' functions to '_Float16'. No back-clone is created
//    since there is no CloneOriginal entry for 'half'.
//
//    Rename:
//        _Z16__spirv_ocl_fabsDh -> _Z16__spirv_ocl_fabsDF16_
//        (half)                 -> (_Float16)
//
// 4. long <-> int (32-bit platforms only)
//    On 32-bit platforms and Windows (LLP64), 'long' is 32-bit.
//    Rename still maps long to long long (see section 1).
//    CloneOriginal additionally clones from 'int' to recreate the 'long'
//    variant.
//
//    Rename:
//        _Z17__spirv_ocl_s_absl -> _Z17__spirv_ocl_s_absx
//        (long)                 -> (long long)
//
//    CloneOriginal (clone, L32):
//        _Z17__spirv_ocl_s_absi -> _Z17__spirv_ocl_s_absl
//        (int)                  -> (long)
//
// 5. Vector Types
//    Vector transformations apply element-type transformations recursively.
//
//    Example: _Z16__spirv_ocl_fabsDv16_Dh -> _Z16__spirv_ocl_fabsDv16_DF16_
//             (vector<half, 16>)          -> (vector<_Float16, 16>)
//
// The implementation includes:
// 1. SPIR type system
// 2. SPIR name mangler for generating mangled names with transformed types
// 3. Bridge from itanium_demangle::Node to SPIR::ParamType
// 4. Type transformation logic for target-specific mappings
// 5. Function cloning with temporary suffix collision handling
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLRemangleLibspirv.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLLowerIR/MangleUtils.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "sycl-remangle-libspirv"

static cl::opt<bool> RemangleSPIRVTarget(
    "remangle-spirv-target",
    cl::desc(
        "Enable SYCL builtin remangling for SPIR/SPIRV targets "
        "(default: false, as these targets already have correct mangling)"),
    cl::init(false));

static cl::opt<unsigned>
    LongWidth("remangle-long-width",
              cl::desc("Width of long type in bits (default: 64)"),
              cl::init(64));

enum class Signedness { Signed, Unsigned };

static cl::opt<Signedness> CharSignedness(
    "remangle-char-signedness",
    cl::values(clEnumValN(Signedness::Signed, "signed", "char is signed."),
               clEnumValN(Signedness::Unsigned, "unsigned",
                          "char is unsigned.")));

static constexpr StringRef TmpSuffix = ".tmp";

namespace {

//===----------------------------------------------------------------------===//
// Extended from SPIR Type System (adapted from SPIRV-LLVM-Translator Mangler).
//===----------------------------------------------------------------------===//

// Transformation mode for type replacements.
enum class TransformMode {
  // Rename (ParameterTypeReplacements): rename OpenCL C types to SYCL types.
  // char -> signed char (always), half -> _Float16
  // long -> long long, unsigned long -> unsigned long long
  Rename,

  // CloneTypeReplacements: find clonee source for creating back-clones.
  // char -> signed char (signed target) or unsigned char (unsigned target)
  // long -> int (L32) or long long (L64)
  // unsigned long -> unsigned int (L32) or unsigned long long (L64)
  // (half, _Float16, signed char, unsigned char: no transformation)
  CloneOriginal,

  // CloneRemangled (RemangledCloneTypeReplacements): find target for second
  // clone. Derived from Rename entries where Rename[k] != CloneOriginal[k],
  // plus char -> signed char.
  // char -> signed char (always), half -> _Float16
  // long -> long long (L32 only; L64: no entry, stays as long)
  // unsigned long -> unsigned long long (L32 only; L64: no entry)
  CloneRemangled,
};

using namespace llvm::itanium_demangle;

// Bridge itanium_demangle::Node to SPIR::ParamType.
class NodeToSPIRType {
  unsigned TargetDefaultAddrSpace;
  TransformMode Mode;

public:
  NodeToSPIRType(const Triple &TT, TransformMode Mode) : Mode(Mode) {
    TargetDefaultAddrSpace = (TT.isSPIR() || TT.isSPIRV()) ? 4 : 0;
  }

  SPIR::RefParamType convert(const Node *N) {
    if (!N)
      return SPIR::RefParamType();

    switch (N->getKind()) {
    case Node::KNameType:
      return convertNameType(static_cast<const NameType *>(N));
    case Node::KPointerType:
      return convertPointerType(
          static_cast<const itanium_demangle::PointerType *>(N));
    case Node::KVectorType:
      return convertVectorType(
          static_cast<const itanium_demangle::VectorType *>(N));
    case Node::KQualType:
      return convert(static_cast<const QualType *>(N)->getChild());
    case Node::KTemplateParamQualifiedArg:
      return convertTemplateParamQualifiedArg(
          static_cast<const TemplateParamQualifiedArg *>(N));
    case Node::KSyntheticTemplateParamName:
      return convertSyntheticTemplateParamName(
          static_cast<const SyntheticTemplateParamName *>(N));
    case Node::KForwardTemplateReference:
      return convertForwardTemplateReference(
          static_cast<const ForwardTemplateReference *>(N));
    case Node::KVendorExtQualType:
      return convertVendorExtQualType(
          static_cast<const VendorExtQualType *>(N));
    case Node::KBinaryFPType:
      return SPIR::RefParamType(
          new SPIR::PrimitiveType(SPIR::PRIMITIVE_FLOAT16));
    default:
      return SPIR::RefParamType();
    }
  }

private:
  SPIR::RefParamType
  convertTemplateParamQualifiedArg(const TemplateParamQualifiedArg *TPQA) {
    const Node *Arg = nullptr;
    TPQA->match([&](Node *, Node *ArgNode) { Arg = ArgNode; });
    return convert(Arg);
  }

  SPIR::RefParamType
  convertSyntheticTemplateParamName(const SyntheticTemplateParamName *TPN) {
    TemplateParamKind Kind = TemplateParamKind::Type;
    unsigned Index = 0;
    TPN->match([&](TemplateParamKind K, unsigned I) {
      Kind = K;
      Index = I;
    });
    if (Kind != TemplateParamKind::Type)
      return SPIR::RefParamType();
    return SPIR::RefParamType(new SPIR::TemplateParameterType(Index));
  }

  SPIR::RefParamType
  convertForwardTemplateReference(const ForwardTemplateReference *FTR) {
    if (FTR->Ref)
      return convert(FTR->Ref);
    return SPIR::RefParamType(new SPIR::TemplateParameterType(FTR->Index));
  }

  // Convert a demangled type name to SPIR type representation.
  // Primitive types are handled explicitly for type transformation. All other
  // types (images, events, samplers, pipes, user structs) become
  // UserDefinedType and pass through unchanged.
  SPIR::RefParamType convertNameType(const NameType *NT) {
    StringRef Name(NT->getName());
    SPIR::TypePrimitiveEnum Prim = SPIR::PRIMITIVE_INT;

    if (Name == "bool")
      Prim = SPIR::PRIMITIVE_BOOL;
    else if (Name == "char") {
      if (Mode == TransformMode::Rename ||
          Mode == TransformMode::CloneRemangled) {
        // Rename and CloneRemangled: always char -> signed char.
        Prim = SPIR::PRIMITIVE_SCHAR;
      } else {
        // CloneOriginal.
        Prim = (CharSignedness == Signedness::Signed) ? SPIR::PRIMITIVE_SCHAR
                                                      : SPIR::PRIMITIVE_UCHAR;
      }
    } else if (Name == "signed char")
      Prim = SPIR::PRIMITIVE_SCHAR;
    else if (Name == "unsigned char")
      Prim = SPIR::PRIMITIVE_UCHAR;
    else if (Name == "short")
      Prim = SPIR::PRIMITIVE_SHORT;
    else if (Name == "unsigned short")
      Prim = SPIR::PRIMITIVE_USHORT;
    else if (Name == "int")
      Prim = SPIR::PRIMITIVE_INT;
    else if (Name == "unsigned int")
      Prim = SPIR::PRIMITIVE_UINT;
    else if (Name == "long") {
      if (Mode == TransformMode::Rename) {
        // Rename: long -> long long (always).
        Prim = SPIR::PRIMITIVE_LONGLONG;
      } else if (Mode == TransformMode::CloneOriginal) {
        // CloneOriginal: long -> int (L32) or long long (L64).
        Prim =
            (LongWidth == 32) ? SPIR::PRIMITIVE_INT : SPIR::PRIMITIVE_LONGLONG;
      } else {
        // CloneRemangled: long -> long long (L32); stays as long on L64.
        Prim =
            (LongWidth == 32) ? SPIR::PRIMITIVE_LONGLONG : SPIR::PRIMITIVE_LONG;
      }
    } else if (Name == "unsigned long") {
      if (Mode == TransformMode::Rename) {
        // Rename: unsigned long -> unsigned long long (always).
        Prim = SPIR::PRIMITIVE_ULONGLONG;
      } else if (Mode == TransformMode::CloneOriginal) {
        // CloneOriginal: unsigned long -> unsigned int (L32) or ull (L64).
        Prim = (LongWidth == 32) ? SPIR::PRIMITIVE_UINT
                                 : SPIR::PRIMITIVE_ULONGLONG;
      } else {
        // CloneRemangled: unsigned long -> ull (L32); stays as ulong on L64.
        Prim = (LongWidth == 32) ? SPIR::PRIMITIVE_ULONGLONG
                                 : SPIR::PRIMITIVE_ULONG;
      }
    } else if (Name == "long long")
      Prim = SPIR::PRIMITIVE_LONGLONG;
    else if (Name == "unsigned long long")
      Prim = SPIR::PRIMITIVE_ULONGLONG;
    else if (Name == "_Float16")
      Prim = SPIR::PRIMITIVE_FLOAT16;
    else if (Name == "half") {
      if (Mode == TransformMode::Rename ||
          Mode == TransformMode::CloneRemangled) {
        // Rename and CloneRemangled: half -> _Float16.
        Prim = SPIR::PRIMITIVE_FLOAT16;
      } else {
        // CloneOriginal: stays as half.
        Prim = SPIR::PRIMITIVE_HALF;
      }
    } else if (Name == "float")
      Prim = SPIR::PRIMITIVE_FLOAT;
    else if (Name == "double")
      Prim = SPIR::PRIMITIVE_DOUBLE;
    else if (Name == "void")
      Prim = SPIR::PRIMITIVE_VOID;
    else
      return SPIR::RefParamType(new SPIR::UserDefinedType(Name));

    return SPIR::RefParamType(new SPIR::PrimitiveType(Prim));
  }

  SPIR::RefParamType
  convertPointerType(const itanium_demangle::PointerType *PT) {
    const Node *PointeeNode = PT->getPointee();

    // Extract address space qualifier from pointer type.
    //
    // In Itanium mangling, address space qualifiers are encoded as vendor
    // extensions in the pointee type, not the pointer itself. We extract
    // the AS qualifier here and apply it to the pointer (which is how SPIR
    // represents it), then continue processing the unwrapped pointee type.
    //
    // We must extract address space BEFORE CV-qualifiers because the structure
    // is: PointerType -> VendorExtQualType(AS) -> QualType(CV) -> BaseType.

    // Default value, e.g. for implicit pointers (no AS qualifier).
    SPIR::TypeAttributeEnum AS =
        (TargetDefaultAddrSpace == 4) ? SPIR::ATTR_PRIVATE : SPIR::ATTR_GENERIC;
    if (PointeeNode->getKind() == Node::KVendorExtQualType) {
      const auto *VT = static_cast<const VendorExtQualType *>(PointeeNode);
      StringRef Ext(VT->getExt());
      if (Ext.starts_with("AS")) {
        int ASNum = 0;
        [[maybe_unused]] bool Error = Ext.drop_front(2).getAsInteger(10, ASNum);
        assert(!Error && "Unexpected non-integer address space");
        if (ASNum == 0)
          AS = SPIR::ATTR_PRIVATE;
        else if (ASNum == 1)
          AS = SPIR::ATTR_GLOBAL;
        else if (ASNum == 2)
          AS = SPIR::ATTR_CONSTANT;
        else if (ASNum == 3)
          AS = SPIR::ATTR_LOCAL;
        else if (ASNum == 4)
          AS = SPIR::ATTR_GENERIC_EXPLICIT;
        else
          llvm_unreachable("Unexpected address space number in mangled name");
        // Use the inner type, not the VendorExtQualType wrapper.
        PointeeNode = VT->getTy();
      }
    }

    // Extract CV-qualifiers (const/volatile/restrict) from QualType nodes.
    // In Itanium mangling, qualifiers appear as QualType wrappers around the
    // actual pointee type. We need to extract and preserve these qualifiers
    // for correct OpenCL function signature matching.
    //
    // Example: pointer to const int with address space is represented as:
    //   PointerType -> VendorExtQualType(AS1) -> QualType(const) ->
    //   NameType("int")
    // And mangles as "PU3AS1Ki" where U3AS1=AS1, K=const, i=int.
    bool IsConst = false;
    bool IsVolatile = false;
    bool IsRestrict = false;

    while (PointeeNode->getKind() == Node::KQualType) {
      const auto *QT = static_cast<const QualType *>(PointeeNode);
      Qualifiers Quals = QT->getQuals();
      if (Quals & QualConst)
        IsConst = true;
      if (Quals & QualVolatile)
        IsVolatile = true;
      if (Quals & QualRestrict)
        IsRestrict = true;
      PointeeNode = QT->getChild();
    }

    SPIR::RefParamType Pointee = convert(PointeeNode);
    if (!Pointee)
      return SPIR::RefParamType();

    auto *Ptr = new SPIR::PointerType(Pointee);
    Ptr->setAddressSpace(AS);
    // Apply CV-qualifiers to the pointer.
    Ptr->setQualifier(SPIR::ATTR_CONST, IsConst);
    Ptr->setQualifier(SPIR::ATTR_VOLATILE, IsVolatile);
    Ptr->setQualifier(SPIR::ATTR_RESTRICT, IsRestrict);
    return SPIR::RefParamType(Ptr);
  }

  SPIR::RefParamType convertVectorType(const itanium_demangle::VectorType *VT) {
    const Node *BaseNode = VT->getBaseType();
    SPIR::RefParamType Base = convert(BaseNode);
    if (!Base)
      return SPIR::RefParamType();

    // Extract vector length from dimension node.
    const Node *DimNode = VT->getDimension();
    if (DimNode->getKind() == Node::KNameType) {
      StringRef DimStr(static_cast<const NameType *>(DimNode)->getName());
      int Len;
      if (!DimStr.getAsInteger(10, Len))
        return SPIR::RefParamType(new SPIR::VectorType(Base, Len));
    }
    return SPIR::RefParamType();
  }

  SPIR::RefParamType convertVendorExtQualType(const VendorExtQualType *VT) {
    StringRef Ext(VT->getExt());
    SPIR::RefParamType Base = convert(VT->getTy());
    if (!Base)
      return Base;

    // Handle address space qualifiers.
    if (Ext.starts_with("AS")) {
      int AS;
      if (!Ext.drop_front(2).getAsInteger(10, AS)) {
        if (auto *PT = dyn_cast<SPIR::PointerType>(Base.get())) {
          SPIR::TypeAttributeEnum Attr = SPIR::ATTR_PRIVATE;
          if (AS == 1)
            Attr = SPIR::ATTR_GLOBAL;
          else if (AS == 2)
            Attr = SPIR::ATTR_CONSTANT;
          else if (AS == 3)
            Attr = SPIR::ATTR_LOCAL;
          else if (AS == 4)
            Attr = SPIR::ATTR_GENERIC_EXPLICIT;
          PT->setAddressSpace(Attr);
        }
      }
    }
    return Base;
  }
};

SPIR::RefParamType cloneType(SPIR::RefParamType Type) {
  if (!Type)
    return Type;

  if (const auto *Prim = dyn_cast<SPIR::PrimitiveType>(Type.get())) {
    // Primitive remapping happens in NodeToSPIRType; this only deep-copies.
    return SPIR::RefParamType(new SPIR::PrimitiveType(Prim->getPrimitive()));
  } else if (const auto *Ptr = dyn_cast<SPIR::PointerType>(Type.get())) {
    SPIR::RefParamType Pointee = cloneType(Ptr->getPointee());
    auto *NewPtr = new SPIR::PointerType(Pointee);
    NewPtr->setAddressSpace(Ptr->getAddressSpace());
    NewPtr->setQualifier(SPIR::ATTR_CONST, Ptr->hasQualifier(SPIR::ATTR_CONST));
    NewPtr->setQualifier(SPIR::ATTR_VOLATILE,
                         Ptr->hasQualifier(SPIR::ATTR_VOLATILE));
    NewPtr->setQualifier(SPIR::ATTR_RESTRICT,
                         Ptr->hasQualifier(SPIR::ATTR_RESTRICT));
    return SPIR::RefParamType(NewPtr);
  } else if (const auto *Vec = dyn_cast<SPIR::VectorType>(Type.get())) {
    SPIR::RefParamType Scalar = cloneType(Vec->getScalarType());
    return SPIR::RefParamType(new SPIR::VectorType(Scalar, Vec->getLength()));
  }
  return Type;
}

// Simple allocator for demangling.
class SimpleAllocator {
  BumpPtrAllocator Alloc;

public:
  void reset() { Alloc.Reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&...A) {
    return new (Alloc.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(A)...);
  }

  void *allocateNodeArray(size_t Sz) {
    return Alloc.Allocate(sizeof(Node *) * Sz, alignof(Node *));
  }
};

using Demangler = ManglingParser<SimpleAllocator>;

struct NameBoundaryParser : ManglingParser<SimpleAllocator> {
  using ManglingParser<SimpleAllocator>::ManglingParser;

  NameBoundaryParser(const char *First, const char *Last,
                     const char *InputBegin)
      : ManglingParser<SimpleAllocator>(First, Last), InputBegin(InputBegin) {}

  size_t getTemplateSuffixStart() const { return First - InputBegin; }

  const char *InputBegin;
};

size_t findTemplateSuffixStart(StringRef MangledName) {
  if (!MangledName.starts_with("_Z"))
    return StringRef::npos;

  NameBoundaryParser Parser(MangledName.data() + 2,
                            MangledName.data() + MangledName.size(),
                            MangledName.data());
  const Node *ParsedName = Parser.parseName();
  if (!ParsedName || ParsedName->getKind() != Node::KNameWithTemplateArgs)
    return StringRef::npos;
  return Parser.getTemplateSuffixStart();
}

bool isValidRemangledBuiltinName(StringRef MangledName,
                                 StringRef ExpectedBaseName) {
  Demangler D(MangledName.data(), MangledName.data() + MangledName.size());
  const Node *Root = D.parse();
  if (!Root || Root->getKind() != Node::KFunctionEncoding)
    return false;

  const auto *Encoding = static_cast<const FunctionEncoding *>(Root);
  const Node *NameNode = Encoding->getName();
  return NameNode && StringRef(NameNode->getBaseName()) == ExpectedBaseName;
}

// Remangle a function name.
// Returns true and writes the new mangled name if transformation is needed.
bool tryRemangleFuncName(StringRef MangledName, const Triple &TT,
                         TransformMode Mode,
                         SmallVectorImpl<char> &RemangledName) {
  RemangledName.clear();

  Demangler D(MangledName.data(), MangledName.data() + MangledName.size());
  const Node *Root = D.parse();
  if (!Root || Root->getKind() != Node::KFunctionEncoding)
    return false;

  const auto *Encoding = static_cast<const FunctionEncoding *>(Root);

  // Check if function has template arguments.
  // For templated functions like __spirv_ImageRead<T>, the structure is:
  //   NameWithTemplateArgs -> Name (base name) + TemplateArgs (template params)
  // For non-templated functions, it's just a Name node.
  const Node *NameNode = Encoding->getName();
  StringRef BaseName = NameNode->getBaseName();
  bool HasTemplateArgs = NameNode->getKind() == Node::KNameWithTemplateArgs;

  // Validate no whitespace (would indicate malformed demangling).
  if (BaseName.contains(' '))
    return false;

  // Convert template arguments and function parameter types.
  NodeToSPIRType Converter(TT, Mode);

  SmallVector<SPIR::RefParamType, 2> TemplateArgTypes;
  SmallVector<SPIR::RefParamType, 8> Params;

  if (HasTemplateArgs) {
    const auto *TemplatedName =
        static_cast<const NameWithTemplateArgs *>(NameNode);
    if (!TemplatedName->TemplateArgs ||
        TemplatedName->TemplateArgs->getKind() != Node::KTemplateArgs)
      return false;

    const auto *TemplateArgList =
        static_cast<const itanium_demangle::TemplateArgs *>(
            TemplatedName->TemplateArgs);
    for (const Node *TemplateArgNode : TemplateArgList->getParams()) {
      SPIR::RefParamType TemplateArgType = Converter.convert(TemplateArgNode);
      if (!TemplateArgType)
        return false;
      TemplateArgType = cloneType(TemplateArgType);
      TemplateArgTypes.push_back(TemplateArgType);
    }
  }

  NodeArray ParamNodes = Encoding->getParams();
  if (ParamNodes.empty() && TemplateArgTypes.empty())
    return false;

  for (const Node *ParamNode : ParamNodes) {
    SPIR::RefParamType ParamType = Converter.convert(ParamNode);
    if (!ParamType) {
      // If any parameter fails to convert, this is likely a malformed/invalid
      // mangling. Don't try to remangle it.
      return false;
    }
    ParamType = cloneType(ParamType);
    Params.push_back(ParamType);
  }

  // Remangle using SPIR mangler. Templated functions are only handled
  // structurally when all template arguments can be represented as supported
  // type nodes; otherwise we conservatively skip remangling.
  SmallString<128> Result;
  // Always use implicit generic address space: TargetDefaultAddrSpace=0
  // preserves AS qualifiers as-is.
  SPIR::NameMangler Mangler;
  if (HasTemplateArgs) {
    size_t TemplateSuffixStart = findTemplateSuffixStart(MangledName);
    if (TemplateSuffixStart == StringRef::npos)
      return false;

    SmallString<128> RemangledTemplatePrefix;
    if (Mangler.mangleTemplateName(BaseName, TemplateArgTypes,
                                   RemangledTemplatePrefix) !=
        SPIR::MANGLE_SUCCESS)
      return false;

    Result = RemangledTemplatePrefix;
    Result += MangledName.substr(TemplateSuffixStart);
  } else if (Mangler.mangle(BaseName, Params, Result) != SPIR::MANGLE_SUCCESS) {
    return false;
  }

  // Final validation: ensure the new mangled name is different from the
  // original. If they're the same or if Result is malformed, don't transform.
  if (Result == MangledName)
    return false;

  assert(isValidRemangledBuiltinName(Result, BaseName) &&
         "invalid remangled builtin name");

  RemangledName = Result;
  return true;
}

// Make a clone of a suitable function using the old name if there is a
// type-mapping and the corresponding clonee function exists.
// TransformMode::CloneOriginal: CloneName=OrigName, CloneeName=TypeResult.
// TransformMode::CloneRemangled: CloneName=TypeResult, CloneeName=OrigName.
void createCloneFromMap(const Triple &TT, Module &M, StringRef OrigName,
                        StringRef MangledForRemap, TransformMode Mode) {
  SmallString<128> RemangledName;
  bool HasRemangledName =
      tryRemangleFuncName(MangledForRemap, TT, Mode, RemangledName);

  if (!HasRemangledName)
    RemangledName = MangledForRemap;

  if (StringRef(RemangledName) == OrigName)
    return; // No transformation needed

  SmallString<128> CloneName;
  StringRef CloneeName;
  if (Mode == TransformMode::CloneOriginal) {
    CloneName = OrigName;
    CloneeName = RemangledName;
  } else {
    CloneName = RemangledName;
    CloneeName = OrigName;
  }

  // TmpSuffix: if CloneName already exists, append .tmp (will be resolved
  // in postProcessRemoveTmpSuffix).
  if (M.getFunction(CloneName)) {
    CloneName += TmpSuffix;
    if (M.getFunction(CloneName))
      return; // Both base and .tmp exist, skip
  }

  if (Function *Clonee = M.getFunction(CloneeName)) {
    ValueToValueMapTy VMap;
    Function *Clone = CloneFunction(Clonee, VMap);
    Clone->setName(CloneName);
    LLVM_DEBUG(dbgs() << "clone " << Clone->getName() << " <- "
                      << Clonee->getName() << "\n");
  }
}

// Resolve .tmp collisions.
void postProcessRemoveTmpSuffix(Module &M,
                                const StringSet<> &RenamedFunctions) {
  SmallVector<Function *, 16> TmpFunctions;
  for (Function &F : M)
    if (F.getName().ends_with(TmpSuffix))
      TmpFunctions.push_back(&F);

  SmallVector<Function *, 8> ToErase;
  for (Function *F : TmpFunctions) {
    StringRef FName = F->getName();
    StringRef BaseName = FName.drop_back(TmpSuffix.size());
    if (Function *Existing = M.getFunction(BaseName)) {
      if (RenamedFunctions.count(BaseName)) {
        // BaseName was an original function that got renamed by Rename. Replace
        // it with this .tmp clone (which has more accurate typing).
        Existing->replaceAllUsesWith(
            ConstantPointerNull::get(Existing->getType()));
        Existing->setName("");
        ToErase.push_back(Existing);
      } else {
        // BaseName exists but was not renamed by Rename. This .tmp is
        // redundant. Replace uses of .tmp with Existing and erase.
        F->replaceAllUsesWith(Existing);
        F->eraseFromParent();
        continue;
      }
    }
    // Rename .tmp function to its base name.
    F->setName(BaseName);
  }

  for (Function *F : ToErase)
    F->eraseFromParent();
}

} // anonymous namespace

PreservedAnalyses SYCLRemangleLibspirvPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  // Require explicit configuration via command line options.
  if (LongWidth.getNumOccurrences() == 0 ||
      CharSignedness.getNumOccurrences() == 0)
    return PreservedAnalyses::all();

  const Triple TT(M.getTargetTriple());

  // Skip SPIR/SPIRV targets by default unless explicitly enabled.
  if ((TT.isSPIR() || TT.isSPIRV()) && !RemangleSPIRVTarget)
    return PreservedAnalyses::all();

  bool Changed = false;
  // Track functions whose original names were renamed.
  StringSet<> RenamedFunctions;
  SmallVector<Function *, 16> Worklist;

  for (Function &F : M) {
    // Skip intrinsics and non-mangled names.
    if (F.isIntrinsic() || !F.getName().starts_with("_Z"))
      continue;

    // Skip __clc_ functions. If we remangle them, it should be done in clc
    // library and it requires creating multiple variants of clc libaries,
    // with mangling scheme aligning with the corresponding libspirv remangling.
    if (F.isDeclaration() && F.getName().contains("__clc_"))
      continue;

    // Skip internal/private linkage functions (won't be called externally).
    if (F.hasLocalLinkage())
      continue;

    Worklist.push_back(&F);
  }

  for (Function *F : Worklist) {
    StringRef Name = F->getName();

    SmallString<128> MangledName = Name;

    // Try to change the parameter types in the function name.
    SmallString<128> RemangledName;
    if (!tryRemangleFuncName(MangledName, TT, TransformMode::Rename,
                             RemangledName)) {
      continue;
    }

    RenamedFunctions.insert(MangledName);

    // TmpSuffix: if target already exists, append .tmp.
    if (M.getFunction(RemangledName))
      RemangledName += TmpSuffix;
    if (M.getFunction(RemangledName))
      continue; // Both base and .tmp exist, skip

    Changed = true;

    F->setName(RemangledName);

    LLVM_DEBUG(dbgs() << "remangle " << MangledName << " -> " << RemangledName
                      << "\n");

    // Declarations: rename only, no cloning (they are forward decls of
    // builtins called from other builtins in a single .cl file).
    if (F->isDeclaration())
      continue;

    // Create clone of original function.
    createCloneFromMap(TT, M, MangledName, MangledName,
                       TransformMode::CloneOriginal);

    // Create clone of remangled function.
    createCloneFromMap(TT, M, RemangledName, MangledName,
                       TransformMode::CloneRemangled);
  }

  postProcessRemoveTmpSuffix(M, RenamedFunctions);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
