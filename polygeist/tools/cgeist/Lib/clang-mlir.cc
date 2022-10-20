//===- clang-mlir.cc - Emit MLIR IRs by walking clang AST--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "Attributes.h"
#include "TargetInfo.h"
#include "TypeUtils.h"
#include "utils.h"

#include "mlir/Conversion/SYCLToLLVM/SYCLFuncRegistry.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Assumptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::sycl;
using namespace mlirclang;

static cl::opt<bool>
    memRefFullRank("memref-fullrank", cl::init(false),
                   cl::desc("Get the full rank of the memref."));

static cl::opt<bool> memRefABI("memref-abi", cl::init(true),
                               cl::desc("Use memrefs when possible"));

cl::opt<std::string> PrefixABI("prefix-abi", cl::init(""),
                               cl::desc("Prefix for emitted symbols"));

static cl::opt<bool>
    CombinedStructABI("struct-abi", cl::init(true),
                      cl::desc("Use literal LLVM ABI for structs"));

static cl::opt<bool>
    UseOldFunctionType("old-func-type", cl::init(false),
                       cl::desc("Use way to compute the function type"));

constexpr llvm::StringLiteral MLIRASTConsumer::DeviceModuleName;

/******************************************************************************/
/*                             CodeGenUtils                                   */
/******************************************************************************/

mlir::IntegerAttr CodeGenUtils::wrapIntegerMemorySpace(unsigned memorySpace,
                                                       MLIRContext *ctx) {
  return memorySpace ? mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                              memorySpace)
                     : nullptr;
}

bool CodeGenUtils::isAggregateTypeForABI(QualType qt) {
  auto hasScalarEvaluationKind = [](QualType qt) {
    qt = qt.getCanonicalType();
    while (true) {
      switch (qt->getTypeClass()) {
      // We operate on atomic values according to their underlying type.
      case clang::Type::Atomic:
        qt = cast<AtomicType>(qt)->getValueType();
        continue;
      case clang::Type::Builtin:
      case clang::Type::Pointer:
      case clang::Type::BlockPointer:
      case clang::Type::LValueReference:
      case clang::Type::RValueReference:
      case clang::Type::MemberPointer:
      case clang::Type::Vector:
      case clang::Type::ExtVector:
      case clang::Type::ConstantMatrix:
      case clang::Type::FunctionProto:
      case clang::Type::FunctionNoProto:
      case clang::Type::Enum:
      case clang::Type::ObjCObjectPointer:
      case clang::Type::Pipe:
      case clang::Type::BitInt:
        return true;
      default:
        return false;
      }
    }
  };

  return !hasScalarEvaluationKind(qt) || qt->isMemberFunctionPointerType();
}

bool CodeGenUtils::isLLVMStructABI(const RecordDecl *RD, llvm::StructType *ST) {
  if (!CombinedStructABI || RD->isUnion())
    return true;

  if (auto CXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (!CXRD->hasDefinition() || CXRD->getNumVBases())
      return true;
    for (const auto m : CXRD->methods())
      if (m->isVirtualAsWritten() || m->isPure())
        return true;
    for (const auto &Base : CXRD->bases())
      if (Base.getType()->getAsCXXRecordDecl()->isEmpty())
        return true;
  }

  if (ST && !ST->isLiteral() &&
      (ST->getName() == "struct._IO_FILE" ||
       ST->getName() == "class.std::basic_ifstream" ||
       ST->getName() == "class.std::basic_istream" ||
       ST->getName() == "class.std::basic_ostream" ||
       ST->getName() == "class.std::basic_ofstream"))
    return true;

  return false;
}

bool CodeGenUtils::determineNoUndef(QualType qt, CodeGen::CodeGenTypes &Types,
                                    const llvm::DataLayout &DL,
                                    const CodeGen::ABIArgInfo &AI,
                                    bool CheckCoerce) {
  llvm::Type *Ty = Types.ConvertTypeForMem(qt);
  if (AI.getKind() == CodeGen::ABIArgInfo::Indirect)
    return true;
  if (AI.getKind() == CodeGen::ABIArgInfo::Extend)
    return true;
  if (!DL.typeSizeEqualsStoreSize(Ty))
    // TODO: This will result in a modest amount of values not marked noundef
    // when they could be. We care about values that *invisibly* contain undef
    // bits from the perspective of LLVM IR.
    return false;
  if (CheckCoerce && AI.canHaveCoerceToType()) {
    llvm::Type *CoerceTy = AI.getCoerceToType();
    if (llvm::TypeSize::isKnownGT(DL.getTypeSizeInBits(CoerceTy),
                                  DL.getTypeSizeInBits(Ty)))
      // If we're coercing to a type with a greater size than the canonical one,
      // we're introducing new undef bits.
      // Coercing to a type of smaller or equal size is ok, as we know that
      // there's no internal padding (typeSizeEqualsStoreSize).
      return false;
  }
  if (qt->isBitIntType())
    return true;
  if (qt->isReferenceType())
    return true;
  if (qt->isNullPtrType())
    return false;
  if (qt->isMemberPointerType())
    // TODO: Some member pointers are `noundef`, but it depends on the ABI. For
    // now, never mark them.
    return false;
  if (qt->isScalarType()) {
    if (const auto *complex = dyn_cast<clang::ComplexType>(qt))
      return determineNoUndef(complex->getElementType(), Types, DL, AI, false);
    return true;
  }
  if (const auto *vector = dyn_cast<clang::VectorType>(qt))
    return determineNoUndef(vector->getElementType(), Types, DL, AI, false);
  if (const auto *matrix = dyn_cast<clang::MatrixType>(qt))
    return determineNoUndef(matrix->getElementType(), Types, DL, AI, false);
  if (const auto *array = dyn_cast<clang::ArrayType>(qt))
    return determineNoUndef(array->getElementType(), Types, DL, AI, false);

  // TODO: Some structs may be `noundef`, in specific situations.
  return false;
}

mlir::IntegerAttr wrapIntegerMemorySpace(unsigned memorySpace,
                                         MLIRContext *ctx) {
  return memorySpace ? mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                              memorySpace)
                     : nullptr;
}

void CodeGenUtils::TypeAndAttrs::getTypes(
    const SmallVectorImpl<TypeAndAttrs> &descriptors,
    SmallVectorImpl<mlir::Type> &types) {
  assert(types.empty() && "Expecting 'types' to be empty");

  for (const TypeAndAttrs &desc : descriptors)
    types.push_back(desc.type);
}

void CodeGenUtils::TypeAndAttrs::getAttributes(
    const SmallVectorImpl<TypeAndAttrs> &descriptors,
    SmallVectorImpl<mlir::NamedAttrList> &attrs) {
  assert(attrs.empty() && "Expecting 'attrs' to be empty");

  for (const TypeAndAttrs &desc : descriptors)
    attrs.push_back(desc.attrs);
}

/******************************************************************************/
/*                               MLIRScanner                                  */
/******************************************************************************/

MLIRScanner::MLIRScanner(MLIRASTConsumer &Glob,
                         mlir::OwningOpRef<mlir::ModuleOp> &module,
                         LowerToInfo &LTInfo)
    : Glob(Glob), function(), module(module), builder(module->getContext()),
      loc(builder.getUnknownLoc()), entryBlock(nullptr), loops(),
      allocationScope(nullptr), supportedFuncs(), bufs(), constants(), labels(),
      EmittingFunctionDecl(nullptr), params(), Captures(), CaptureKinds(),
      ThisCapture(nullptr), arrayinit(), ThisVal(), returnVal(),
      LTInfo(LTInfo) {}

void MLIRScanner::initSupportedFunctions() {
  // Functions needed for single_task with one dimensional write buffer.

  // SYCL constructors:
  supportedFuncs.insert("_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_"
                        "2idILi1EEENS0_5rangeILi1EEES7_");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEEC1Ev");
  supportedFuncs.insert("_ZN4sycl3_V16detail5arrayILi1EEC1ILi1EEENSt9enable_"
                        "ifIXeqT_Li1EEmE4typeE");
  supportedFuncs.insert("_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_");
  supportedFuncs.insert("_ZN4sycl3_V12idILi1EEC1Ev");
  supportedFuncs.insert("_ZN4sycl3_V12idILi1EEC1ERKS2_");
  supportedFuncs.insert(
      "_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE");
  supportedFuncs.insert("_ZN4sycl3_V15rangeILi1EEC1ERKS2_");
  supportedFuncs.insert(
      "_ZN4sycl3_V15rangeILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE");

  // Other SYCL functions:
  supportedFuncs.insert(
      "_ZN4sycl3_V13ext6oneapi22accessor_property_listIJEE12has_propertyINS2_"
      "8property9no_offsetEEEbPNSt9enable_ifIXsr24is_compile_time_propertyIT_"
      "EE5valueEvE4typeE");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEEixILi1EvEERiNS0_2idILi1EEE");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getAccessRangeEv");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE");
  supportedFuncs.insert(
      "_ZZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEEENKUlmE_clEm");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getMemoryRangeEv");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getMemoryRangeEv");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE9getOffsetEv");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getTotalOffsetEv");
  supportedFuncs.insert(
      "_ZZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getTotalOffsetEvENKUlmE_clEm");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE15getQualifiedPtrEv");
  supportedFuncs.insert("_ZN4sycl3_V16detail5arrayILi1EEixEi");
  supportedFuncs.insert("_ZNK4sycl3_V16detail5arrayILi1EEixEi");
  supportedFuncs.insert("_ZNK4sycl3_V16detail5arrayILi1EE15check_dimensionEi");
  supportedFuncs.insert("_ZN4sycl3_V16detail14InitializedValILi1ENS0_"
                        "5rangeEE3getILi0EEENS3_ILi1EEEv");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail8dim_loopILm1EZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_"
      "2idIXT_EEEEUlmE_EEvOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail8dim_loopILm1EZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getTotalOffsetEvEUlmE_"
      "EEvOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail13dim_loop_implIJLm0EEZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getTotalOffsetEvEUlmE_"
      "EEvSt16integer_sequenceImJXspT_EEEOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail13dim_loop_implIJLm0EEZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_"
      "2idIXT_EEEEUlmE_EEvSt16integer_sequenceImJXspT_EEEOT0_");
}

static void checkFunctionParent(const FunctionOpInterface F,
                                FunctionContext Context,
                                const OwningOpRef<ModuleOp> &module) {
  assert(
      (Context != FunctionContext::Host || F->getParentOp() == module.get()) &&
      "New function must be inserted into global module");
  assert((Context != FunctionContext::SYCLDevice ||
          F->getParentOfType<gpu::GPUModuleOp>() == getDeviceModule(*module)) &&
         "New device function must be inserted into device module");
}

static constexpr unsigned AllocSizeNumElemsNotPresent = -1;

static uint64_t packAllocSizeArgs(unsigned ElemSizeArg,
                                  const Optional<unsigned> &NumElemsArg) {
  assert((!NumElemsArg || *NumElemsArg != AllocSizeNumElemsNotPresent) &&
         "Attempting to pack a reserved value");

  return uint64_t(ElemSizeArg) << 32 |
         NumElemsArg.value_or(AllocSizeNumElemsNotPresent);
}

static void
addAttributesFromFunctionProtoType(mlirclang::AttrBuilder &funcAttrs,
                                   const clang::FunctionProtoType *FPT) {
  if (!FPT)
    return;

  if (!isUnresolvedExceptionSpec(FPT->getExceptionSpecType()) &&
      FPT->isNothrow())
    funcAttrs.addAttribute(llvm::Attribute::NoUnwind);
}

static void addAttributesFromAssumes(mlirclang::AttrBuilder &funcAttrs,
                                     const Decl *callee) {
  if (!callee)
    return;

  SmallVector<StringRef, 4> attrs;
  for (const AssumptionAttr *AA : callee->specific_attrs<AssumptionAttr>())
    AA->getAssumption().split(attrs, ",");

  MLIRContext *ctx = &funcAttrs.getContext();
  SmallVector<mlir::Attribute, 4> strAttrs;
  for (StringRef attrName : attrs)
    strAttrs.push_back(StringAttr::get(ctx, attrName));

  funcAttrs.addAttribute(llvm::AssumptionAttrKey,
                         mlir::ArrayAttr::get(ctx, strAttrs));
}

static void addNoBuiltinAttributes(mlirclang::AttrBuilder &funcAttrs,
                                   const LangOptions &langOpts,
                                   const NoBuiltinAttr *NBA = nullptr) {
  auto addNoBuiltinAttr = [&funcAttrs](StringRef builtinName) {
    SmallString<32> attributeName;
    attributeName += "no-builtin-";
    attributeName += builtinName;
    funcAttrs.addAttribute(attributeName, mlir::UnitAttr());
  };

  // First, handle the language options passed through -fno-builtin.
  if (langOpts.NoBuiltin) {
    // -fno-builtin disables them all.
    funcAttrs.addAttribute("no-builtins", mlir::UnitAttr());
    return;
  }

  // Then, add attributes for builtins specified through -fno-builtin-<name>.
  llvm::for_each(langOpts.NoBuiltinFuncs, addNoBuiltinAttr);

  // Now, let's check the __attribute__((no_builtin("...")) attribute added to
  // the source.
  if (!NBA)
    return;

  // If there is a wildcard in the builtin names specified through the
  // attribute, disable them all.
  if (llvm::is_contained(NBA->builtinNames(), "*")) {
    funcAttrs.addAttribute("no-builtins", mlir::UnitAttr());
    return;
  }

  // And last, add the rest of the builtin names.
  llvm::for_each(NBA->builtinNames(), addNoBuiltinAttr);
}

namespace {
// ABIArgInfo::Expand implementation.

// Specifies the way QualType passed as ABIArgInfo::Expand is expanded.
struct TypeExpansion {
  enum TypeExpansionKind {
    // Elements of constant arrays are expanded recursively.
    TEK_ConstantArray,
    // Record fields are expanded recursively (but if record is a union, only
    // the field with the largest size is expanded).
    TEK_Record,
    // For complex types, real and imaginary parts are expanded recursively.
    TEK_Complex,
    // All other types are not expandable.
    TEK_None
  };

  const TypeExpansionKind Kind;

  TypeExpansion(TypeExpansionKind K) : Kind(K) {}
  virtual ~TypeExpansion() {}
};

struct ConstantArrayExpansion : TypeExpansion {
  QualType EltTy;
  uint64_t NumElts;

  ConstantArrayExpansion(QualType EltTy, uint64_t NumElts)
      : TypeExpansion(TEK_ConstantArray), EltTy(EltTy), NumElts(NumElts) {}
  static bool classof(const TypeExpansion *TE) {
    return TE->Kind == TEK_ConstantArray;
  }
};

struct RecordExpansion : TypeExpansion {
  SmallVector<const CXXBaseSpecifier *, 1> Bases;

  SmallVector<const FieldDecl *, 1> Fields;

  RecordExpansion(SmallVector<const CXXBaseSpecifier *, 1> &&Bases,
                  SmallVector<const FieldDecl *, 1> &&Fields)
      : TypeExpansion(TEK_Record), Bases(std::move(Bases)),
        Fields(std::move(Fields)) {}
  static bool classof(const TypeExpansion *TE) {
    return TE->Kind == TEK_Record;
  }
};

struct ComplexExpansion : TypeExpansion {
  QualType EltTy;

  ComplexExpansion(QualType EltTy) : TypeExpansion(TEK_Complex), EltTy(EltTy) {}
  static bool classof(const TypeExpansion *TE) {
    return TE->Kind == TEK_Complex;
  }
};

struct NoExpansion : TypeExpansion {
  NoExpansion() : TypeExpansion(TEK_None) {}
  static bool classof(const TypeExpansion *TE) { return TE->Kind == TEK_None; }
};
} // namespace

static std::unique_ptr<TypeExpansion>
getTypeExpansion(QualType Ty, const ASTContext &Context) {
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty)) {
    return std::make_unique<ConstantArrayExpansion>(
        AT->getElementType(), AT->getSize().getZExtValue());
  }
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    SmallVector<const CXXBaseSpecifier *, 1> Bases;
    SmallVector<const FieldDecl *, 1> Fields;
    const RecordDecl *RD = RT->getDecl();
    assert(!RD->hasFlexibleArrayMember() &&
           "Cannot expand structure with flexible array.");
    if (RD->isUnion()) {
      // Unions can be here only in degenerative cases - all the fields are same
      // after flattening. Thus we have to use the "largest" field.
      const FieldDecl *LargestFD = nullptr;
      CharUnits UnionSize = CharUnits::Zero();

      for (const auto *FD : RD->fields()) {
        if (FD->isZeroLengthBitField(Context))
          continue;
        assert(!FD->isBitField() &&
               "Cannot expand structure with bit-field members.");
        CharUnits FieldSize = Context.getTypeSizeInChars(FD->getType());
        if (UnionSize < FieldSize) {
          UnionSize = FieldSize;
          LargestFD = FD;
        }
      }
      if (LargestFD)
        Fields.push_back(LargestFD);
    } else {
      if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
        assert(!CXXRD->isDynamicClass() &&
               "cannot expand vtable pointers in dynamic classes");
        llvm::append_range(Bases, llvm::make_pointer_range(CXXRD->bases()));
      }

      for (const auto *FD : RD->fields()) {
        if (FD->isZeroLengthBitField(Context))
          continue;
        assert(!FD->isBitField() &&
               "Cannot expand structure with bit-field members.");
        Fields.push_back(FD);
      }
    }
    return std::make_unique<RecordExpansion>(std::move(Bases),
                                             std::move(Fields));
  }
  if (const clang::ComplexType *CT = Ty->getAs<clang::ComplexType>()) {
    return std::make_unique<ComplexExpansion>(CT->getElementType());
  }
  return std::make_unique<NoExpansion>();
}

static int getExpansionSize(QualType Ty, const ASTContext &Context) {
  auto Exp = getTypeExpansion(Ty, Context);
  if (auto CAExp = dyn_cast<ConstantArrayExpansion>(Exp.get())) {
    return CAExp->NumElts * getExpansionSize(CAExp->EltTy, Context);
  }
  if (auto RExp = dyn_cast<RecordExpansion>(Exp.get())) {
    int Res = 0;
    for (auto BS : RExp->Bases)
      Res += getExpansionSize(BS->getType(), Context);
    for (auto FD : RExp->Fields)
      Res += getExpansionSize(FD->getType(), Context);
    return Res;
  }
  if (isa<ComplexExpansion>(Exp.get()))
    return 2;
  assert(isa<NoExpansion>(Exp.get()));
  return 1;
}

void MLIRScanner::init(mlir::FunctionOpInterface func,
                       const FunctionToEmit &FTE) {
  const clang::FunctionDecl *FD = &FTE.getDecl();

  function = func;
  EmittingFunctionDecl = FD;

  if (ShowAST)
    llvm::dbgs() << "Emitting fn: " << function.getName() << "\n"
                 << "\tfunctionDecl:" << *FD << "\n";

  initSupportedFunctions();
  // This is needed, as GPUFuncOps are already created with an entry block.
  setEntryAndAllocBlock(isa<gpu::GPUFuncOp>(function)
                            ? &function.getBlocks().front()
                            : function.addEntryBlock());

  unsigned i = 0;
  if (auto CM = dyn_cast<CXXMethodDecl>(FD)) {
    if (CM->getParent()->isLambda()) {
      for (auto C : CM->getParent()->captures()) {
        if (C.capturesVariable())
          CaptureKinds[C.getCapturedVar()] = C.getCaptureKind();
      }

      CM->getParent()->getCaptureFields(Captures, ThisCapture);
      if (ThisCapture) {
        llvm::errs() << " thiscapture:\n";
        ThisCapture->dump();
      }
    }

    if (CM->isInstance()) {
      mlir::Value val = function.getArgument(i);
      ThisVal = ValueCategory(val, /*isReference*/ false);
      i++;
    }
  }

  const bool isKernel = FD->hasAttr<SYCLKernelAttr>();

  for (ParmVarDecl *parm : FD->parameters()) {
    assert(i != function.getNumArguments());

    QualType parmType = parm->getType();

    bool LLVMABI = false, isArray = false;
    if (Glob.getMLIRType(Glob.getCGM().getContext().getPointerType(parmType))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getMLIRType(parmType, &isArray);

    if (!isArray &&
        isa<clang::ReferenceType>(parmType->getUnqualifiedDesugaredType()))
      isArray = true;

    mlir::Value val = function.getArgument(i);
    assert(val && "Expecting a valid value");

    if (isArray || (isKernel && CodeGenUtils::isAggregateTypeForABI(parmType)))
      params.emplace(parm, ValueCategory(val, /*isReference*/ true));
    else {
      mlir::Value alloc =
          createAllocOp(val.getType(), parm, /*memspace*/ 0, isArray, LLVMABI);
      ValueCategory(alloc, /*isReference*/ true).store(builder, val);
    }

    i++;
  }

  if (FD->hasAttr<CUDAGlobalAttr>() && Glob.getCGM().getLangOpts().CUDA &&
      !Glob.getCGM().getLangOpts().CUDAIsDevice) {
    FunctionToEmit FTE(*FD);
    auto deviceStub = cast<func::FuncOp>(Glob.GetOrCreateMLIRFunction(
        FTE, false /* IsThink*/, true /* ShouldEmit*/,
        /* getDeviceStub */ true));

    builder.create<func::CallOp>(loc, deviceStub, function.getArguments());
    builder.create<ReturnOp>(loc);
    return;
  }

  if (auto CC = dyn_cast<CXXConstructorDecl>(FD)) {
    const CXXRecordDecl *ClassDecl = CC->getParent();
    for (auto expr : CC->inits()) {
      if (ShowAST) {
        llvm::errs() << " init: - baseInit:" << (int)expr->isBaseInitializer()
                     << " memberInit:" << (int)expr->isMemberInitializer()
                     << " anyMember:" << (int)expr->isAnyMemberInitializer()
                     << " indirectMember:"
                     << (int)expr->isIndirectMemberInitializer()
                     << " isinClass:" << (int)expr->isInClassMemberInitializer()
                     << " delegating:" << (int)expr->isDelegatingInitializer()
                     << " isPack:" << (int)expr->isPackExpansion() << "\n";
        if (expr->getMember())
          expr->getMember()->dump();
        if (expr->getInit())
          expr->getInit()->dump();
      }
      assert(ThisVal.val);

      FieldDecl *field = expr->getMember();
      if (!field) {
        if (expr->isBaseInitializer()) {
          bool BaseIsVirtual = expr->isBaseVirtual();
          auto BaseType = expr->getBaseClass();

          // Shift and cast down to the base type.
          // TODO: for complete types, this should be possible with a GEP.
          mlir::Value V = ThisVal.val;
          const clang::Type *BaseTypes[] = {BaseType};
          bool BaseVirtual[] = {BaseIsVirtual};

          V = GetAddressOfBaseClass(V, /*derived*/ ClassDecl, BaseTypes,
                                    BaseVirtual);

          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0, /*mem*/ V);
          continue;
        }

        if (expr->isDelegatingInitializer()) {
          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0,
                               /*mem*/ ThisVal.val);
          continue;
        }
      }
      assert(field && "initialiation expression must apply to a field");

      if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getInit())) {
        VisitArrayInitLoop(AILE,
                           CommonFieldLookup(CC->getThisObjectType(), field,
                                             ThisVal.val, /*isLValue*/ false));
        continue;
      }

      if (auto cons = dyn_cast<CXXConstructExpr>(expr->getInit())) {
        VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0,
                             CommonFieldLookup(CC->getThisObjectType(), field,
                                               ThisVal.val, /*isLValue*/ false)
                                 .val);
        continue;
      }

      auto initexpr = Visit(expr->getInit());
      if (!initexpr.val) {
        expr->getInit()->dump();
        assert(initexpr.val);
      }

      bool isArray = false;
      Glob.getMLIRType(expr->getInit()->getType(), &isArray);

      auto cfl = CommonFieldLookup(CC->getThisObjectType(), field, ThisVal.val,
                                   /*isLValue*/ false);
      assert(cfl.val);
      cfl.store(builder, initexpr, isArray);
    }
  }

  if (auto CC = dyn_cast<CXXDestructorDecl>(FD)) {
    CC->dump();
    llvm::errs() << " warning, destructor not fully handled yet\n";
  }

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().keepRunning);

  if (function.getResultTypes().size()) {
    auto type = mlir::MemRefType::get({}, function.getResultTypes()[0], {}, 0);
    returnVal = builder.create<mlir::memref::AllocaOp>(loc, type);
    if (type.getElementType().isa<mlir::IntegerType, mlir::FloatType>()) {
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::LLVM::UndefOp>(loc, type.getElementType()),
          returnVal, std::vector<mlir::Value>({}));
    }
  }

  if (auto D = dyn_cast<CXXMethodDecl>(FD)) {
    // ClangAST incorrectly does not contain the correct definition
    // of a union move operation and as such we _must_ emit a memcpy
    // for a defaulted union copy or move.
    if (D->getParent()->isUnion() && D->isDefaulted()) {
      mlir::Value V = ThisVal.val;
      assert(V);
      if (auto MT = V.getType().dyn_cast<MemRefType>())
        V = builder.create<polygeist::Pointer2MemrefOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), V);

      mlir::Value src = function.getArgument(1);
      if (auto MT = src.getType().dyn_cast<MemRefType>())
        src = builder.create<polygeist::Pointer2MemrefOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), src);

      mlir::Value typeSize = builder.create<polygeist::TypeSizeOp>(
          loc, builder.getIndexType(),
          mlir::TypeAttr::get(
              V.getType().cast<LLVM::LLVMPointerType>().getElementType()));
      typeSize = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    typeSize);
      V = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              V.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          V);
      src = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              src.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          src);
      mlir::Value volatileCpy = builder.create<ConstantIntOp>(loc, false, 1);
      builder.create<LLVM::MemcpyOp>(loc, V, src, typeSize, volatileCpy);
    }
  }

  Stmt *stmt = FD->getBody();
  assert(stmt);
  if (ShowAST)
    stmt->dump();

  Visit(stmt);

  if (function.getResultTypes().size()) {
    assert(!isa<gpu::GPUFuncOp>(function) &&
           "SYCL kernel functions must always have a void return type.");
    mlir::Value vals[1] = {
        builder.create<mlir::memref::LoadOp>(loc, returnVal)};
    builder.create<ReturnOp>(loc, vals);
  } else if (isa<gpu::GPUFuncOp>(function))
    builder.create<gpu::ReturnOp>(loc);
  else
    builder.create<ReturnOp>(loc);

  checkFunctionParent(function, FTE.getContext(), module);
}

mlir::Value MLIRScanner::createAllocOp(mlir::Type t, VarDecl *name,
                                       uint64_t memspace, bool isArray = false,
                                       bool LLVMABI = false) {

  mlir::MemRefType mr;
  mlir::Value alloc = nullptr;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto varLoc = name ? getMLIRLocation(name->getBeginLoc()) : loc;
  if (!isArray) {
    if (LLVMABI) {
      if (name)
        if (auto var = dyn_cast<VariableArrayType>(
                name->getType()->getUnqualifiedDesugaredType())) {
          auto len = Visit(var->getSizeExpr()).getValue(builder);
          alloc = builder.create<mlir::LLVM::AllocaOp>(
              varLoc, LLVM::LLVMPointerType::get(t, memspace), len);
          builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
          alloc = builder.create<mlir::LLVM::BitcastOp>(
              varLoc,
              LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 0)),
              alloc);
        }

      if (!alloc) {
        alloc = abuilder.create<mlir::LLVM::AllocaOp>(
            varLoc, mlir::LLVM::LLVMPointerType::get(t, memspace),
            abuilder.create<arith::ConstantIntOp>(varLoc, 1, 64), 0);
        if (t.isa<mlir::IntegerType, mlir::FloatType>()) {
          abuilder.create<LLVM::StoreOp>(
              varLoc, abuilder.create<mlir::LLVM::UndefOp>(varLoc, t), alloc);
        }
        // alloc = builder.create<mlir::LLVM::BitcastOp>(varLoc,
        // LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 1)), alloc);
      }
    } else {
      mr = mlir::MemRefType::get(1, t, {}, memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      LLVM_DEBUG({
        llvm::dbgs() << "MLIRScanner::createAllocOp: created: ";
        alloc.dump();
      });

      if (memspace != 0) {
        // Note: this code is incorrect because 'alloc' has a MemRefType in
        // memory space that is not zero, therefore is illegal to create a
        // Memref2Pointer operation that yields a result not in the same memory
        // space.
        auto memRefToPtr = abuilder.create<polygeist::Memref2PointerOp>(
            varLoc, LLVM::LLVMPointerType::get(t, 0), alloc);
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(-1, t, {}, memspace), memRefToPtr);
      }
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(-1, t, {}, 0), alloc);
      LLVM_DEBUG({
        llvm::dbgs() << "MLIRScanner::createAllocOp: created: ";
        alloc.dump();
        llvm::dbgs() << "\n";
      });

      if (t.isa<mlir::IntegerType, mlir::FloatType>()) {
        mlir::Value idxs[] = {abuilder.create<ConstantIndexOp>(loc, 0)};
        abuilder.create<mlir::memref::StoreOp>(
            varLoc, abuilder.create<mlir::LLVM::UndefOp>(varLoc, t), alloc,
            idxs);
      }
    }
  } else {
    auto mt = t.cast<mlir::MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    auto pshape = shape[0];

    if (name)
      if (auto var = dyn_cast<VariableArrayType>(
              name->getType()->getUnqualifiedDesugaredType())) {
        assert(shape[0] == -1);
        mr = mlir::MemRefType::get(
            shape, mt.getElementType(), MemRefLayoutAttrInterface(),
            CodeGenUtils::wrapIntegerMemorySpace(memspace, mt.getContext()));
        auto len = Visit(var->getSizeExpr()).getValue(builder);
        len = builder.create<IndexCastOp>(varLoc, builder.getIndexType(), len);
        alloc = builder.create<mlir::memref::AllocaOp>(varLoc, mr, len);
        builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
        if (memspace != 0) {
          alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
              varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
              abuilder.create<polygeist::Memref2PointerOp>(
                  varLoc, LLVM::LLVMPointerType::get(mt.getElementType(), 0),
                  alloc));
        }
      }

    if (!alloc) {
      if (pshape == -1)
        shape[0] = 1;
      mr = mlir::MemRefType::get(
          shape, mt.getElementType(), MemRefLayoutAttrInterface(),
          CodeGenUtils::wrapIntegerMemorySpace(memspace, mt.getContext()));
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(mt.getElementType(), 0),
                alloc));
      }
      shape[0] = pshape;
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(shape, mt.getElementType()), alloc);
    }
  }
  assert(alloc);
  // NamedAttribute attrs[] = {NamedAttribute("name", name)};
  if (name) {
    // if (name->getName() == "i")
    //  assert(0 && " not i");
    if (params.find(name) != params.end()) {
      name->dump();
    }
    assert(params.find(name) == params.end());
    params[name] = ValueCategory(alloc, /*isReference*/ true);
  }
  return alloc;
}

ValueCategory MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  decl = decl->getCanonicalDecl();
  mlir::Type subType = getMLIRType(decl->getType());
  ValueCategory inite = nullptr;
  unsigned memtype = decl->hasAttr<CUDASharedAttr>() ? 5 : 0;
  bool LLVMABI = false;
  bool isArray = false;

  if (Glob.getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                           decl->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else
    Glob.getMLIRType(decl->getType(), &isArray);

  if (!LLVMABI && isArray) {
    subType = Glob.getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(decl->getType()));
  }

  if (auto init = decl->getInit()) {
    if (!isa<InitListExpr>(init) && !isa<CXXConstructExpr>(init)) {
      auto visit = Visit(init);
      if (!visit.val) {
        decl->dump();
        assert(visit.val);
      }
      bool isReference = init->isLValue() || init->isXValue();
      if (isReference) {
        assert(visit.isReference);
        builder.create<polygeist::TrivialUseOp>(loc, visit.val);
        return params[decl] = visit;
      }
      if (isArray) {
        assert(visit.isReference);
        inite = visit;
      } else {
        inite = ValueCategory(visit.getValue(builder), /*isRef*/ false);
        if (!inite.val) {
          init->dump();
          assert(0 && inite.val);
        }
        subType = inite.val.getType();
      }
    }
  } else if (auto ava = decl->getAttr<AlignValueAttr>()) {
    if (auto algn = dyn_cast<clang::ConstantExpr>(ava->getAlignment())) {
      for (auto a : algn->children()) {
        if (auto IL = dyn_cast<IntegerLiteral>(a)) {
          if (IL->getValue() == 8192) {
            llvm::Type *T =
                Glob.getCGM().getTypes().ConvertType(decl->getType());
            subType = Glob.getTypeTranslator().translateType(T);
            LLVMABI = true;
            break;
          }
        }
      }
    }
  } else if (auto ava = decl->getAttr<InitPriorityAttr>()) {
    if (ava->getPriority() == 8192) {
      llvm::Type *T = Glob.getCGM().getTypes().ConvertType(decl->getType());
      subType = Glob.getTypeTranslator().translateType(T);
      LLVMABI = true;
    }
  }

  mlir::Value op;

  Block *block = nullptr;
  Block::iterator iter;

  if (decl->isStaticLocal() && memtype == 0) {
    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);
    auto varLoc = getMLIRLocation(decl->getBeginLoc());

    if (Glob.getMLIRType(
                Glob.getCGM().getContext().getPointerType(decl->getType()))
            .isa<mlir::LLVM::LLVMPointerType>()) {
      op = abuilder.create<mlir::LLVM::AddressOfOp>(
          varLoc, Glob.GetOrCreateLLVMGlobal(
                      decl, (function.getName() + "@static@").str()));
    } else {
      auto gv =
          Glob.GetOrCreateGlobal(decl, (function.getName() + "@static@").str(),
                                 /*tryInit*/ false);
      op = abuilder.create<memref::GetGlobalOp>(varLoc, gv.first.type(),
                                                gv.first.getName());
    }
    params[decl] = ValueCategory(op, /*isReference*/ true);
    if (decl->getInit()) {
      auto mr = MemRefType::get({1}, builder.getI1Type());
      bool inits[1] = {true};
      auto rtt = RankedTensorType::get({1}, builder.getI1Type());
      auto init_value = DenseIntElementsAttr::get(rtt, inits);
      OpBuilder gbuilder(builder.getContext());
      gbuilder.setInsertionPointToStart(module->getBody());
      auto name = Glob.getCGM().getMangledName(decl);
      auto globalOp = gbuilder.create<mlir::memref::GlobalOp>(
          module->getLoc(),
          builder.getStringAttr(function.getName() + "@static@" + name +
                                "@init"),
          /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
          init_value, mlir::UnitAttr(), /*alignment*/ nullptr);
      SymbolTable::setSymbolVisibility(globalOp,
                                       mlir::SymbolTable::Visibility::Private);

      auto boolop =
          builder.create<memref::GetGlobalOp>(varLoc, mr, globalOp.getName());
      auto cond = builder.create<memref::LoadOp>(
          varLoc, boolop, std::vector<mlir::Value>({getConstantIndex(0)}));

      auto ifOp = builder.create<scf::IfOp>(varLoc, cond, /*hasElse*/ false);
      block = builder.getInsertionBlock();
      iter = builder.getInsertionPoint();
      builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
      builder.create<memref::StoreOp>(
          varLoc, builder.create<ConstantIntOp>(varLoc, false, 1), boolop,
          std::vector<mlir::Value>({getConstantIndex(0)}));
    }
  } else
    op = createAllocOp(subType, decl, memtype, isArray, LLVMABI);

  if (inite.val) {
    ValueCategory(op, /*isReference*/ true).store(builder, inite, isArray);
  } else if (auto init = decl->getInit()) {
    if (isa<InitListExpr>(init)) {
      InitializeValueByInitListExpr(op, init);
    } else if (auto CE = dyn_cast<CXXConstructExpr>(init)) {
      VisitConstructCommon(CE, decl, memtype, op);
    } else
      assert(0 && "unknown init list");
  }
  if (block)
    builder.setInsertionPoint(block, iter);
  return ValueCategory(op, /*isReference*/ true);
}

mlir::Value add(MLIRScanner &sc, mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value lhs, mlir::Value rhs) {
  assert(lhs);
  assert(rhs);
  if (auto op = lhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = lhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }
  return builder.create<AddIOp>(loc, lhs, rhs);
}

mlir::Value MLIRScanner::castToIndex(mlir::Location loc, mlir::Value val) {
  assert(val && "Expect non-null value");

  if (auto op = val.getDefiningOp<ConstantIntOp>())
    return getConstantIndex(op.value());

  return builder.create<arith::IndexCastOp>(
      loc, mlir::IndexType::get(val.getContext()), val);
}

mlir::Value MLIRScanner::castToMemSpace(mlir::Value val, unsigned memSpace) {
  assert(val && "Expect non-null value");

  return mlir::TypeSwitch<mlir::Type, mlir::Value>(val.getType())
      .Case<mlir::MemRefType>([&](mlir::MemRefType valType) -> mlir::Value {
        if (valType.getMemorySpaceAsInt() == memSpace)
          return val;

        mlir::Value newVal = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(valType.getElementType(),
                                       valType.getMemorySpaceAsInt()),
            val);
        newVal = builder.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(valType.getElementType(), memSpace),
            newVal);
        return builder.create<polygeist::Pointer2MemrefOp>(
            loc,
            mlir::MemRefType::get(valType.getShape(), valType.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  CodeGenUtils::wrapIntegerMemorySpace(
                                      memSpace, valType.getContext())),
            newVal);
      })
      .Case<LLVM::LLVMPointerType>(
          [&](LLVM::LLVMPointerType valType) -> mlir::Value {
            if (valType.getAddressSpace() == memSpace)
              return val;

            return builder.create<LLVM::AddrSpaceCastOp>(
                loc,
                LLVM::LLVMPointerType::get(valType.getElementType(), memSpace),
                val);
          })
      .Default([&](mlir::Type valType) {
        llvm_unreachable("unimplemented");
        return val;
      });
}

mlir::Value MLIRScanner::castToMemSpaceOfType(mlir::Value val, mlir::Type t) {
  assert((t.isa<mlir::MemRefType>() || t.isa<LLVM::LLVMPointerType>()) &&
         "Unexpected type");
  unsigned memSpace = t.isa<mlir::MemRefType>()
                          ? t.cast<mlir::MemRefType>().getMemorySpaceAsInt()
                          : t.cast<LLVM::LLVMPointerType>().getAddressSpace();
  return castToMemSpace(val, memSpace);
}

ValueCategory MLIRScanner::CommonArrayToPointer(ValueCategory scalar) {
  assert(scalar.val);
  assert(scalar.isReference);
  if (auto PT = scalar.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (PT.getElementType().isa<mlir::LLVM::LLVMPointerType>())
      return ValueCategory(scalar.val, /*isRef*/ false);
    mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, 0, 32)};
    if (!PT.getElementType().isa<mlir::LLVM::LLVMArrayType>()) {
      EmittingFunctionDecl->dump();
      function.dump();
      llvm::errs() << " sval: " << scalar.val << "\n";
      llvm::errs() << PT << "\n";
    }
    auto ET =
        PT.getElementType().cast<mlir::LLVM::LLVMArrayType>().getElementType();
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
            scalar.val, vec),
        /*isReference*/ false);
  }

  auto mt = scalar.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  // if (shape.size() > 1) {
  //  shape.erase(shape.begin());
  //} else {
  shape[0] = -1;
  //}
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());

  auto post = builder.create<memref::CastOp>(loc, mt0, scalar.val);
  return ValueCategory(post, /*isReference*/ false);
}

ValueCategory MLIRScanner::CommonArrayLookup(ValueCategory array,
                                             mlir::Value idx,
                                             bool isImplicitRefResult,
                                             bool removeIndex) {
  mlir::Value val = array.getValue(builder);
  assert(val);

  if (val.getType().isa<LLVM::LLVMPointerType>()) {

    mlir::Value vals[] = {
        builder.create<IndexCastOp>(loc, builder.getIntegerType(64), idx)};
    // TODO sub
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), val, vals),
        /*isReference*/ true);
  }
  if (!val.getType().isa<MemRefType>()) {
    EmittingFunctionDecl->dump();
    builder.getInsertionBlock()->dump();
    function.dump();
    llvm::errs() << "value: " << val << "\n";
  }

  ValueCategory dref;
  {
    auto mt = val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, val, idx);
    // TODO sub
    dref = ValueCategory(post, /*isReference*/ true);
  }
  assert(dref.isReference);
  if (!removeIndex)
    return dref;

  auto mt = dref.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() == 1 || (shape.size() == 2 && isImplicitRefResult)) {
    shape[0] = -1;
  } else {
    shape.erase(shape.begin());
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, dref.val,
                                                    getConstantIndex(0));
  return ValueCategory(post, /*isReference*/ true);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitGPUCallExpr(clang::CallExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__syncthreads") {
        builder.create<mlir::NVVM::Barrier0Op>(loc);
        return std::make_pair(ValueCategory(), true);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "cudaFuncSetCacheConfig") {
        llvm::errs() << " Not emitting GPU option: cudaFuncSetCacheConfig\n";
        return std::make_pair(ValueCategory(), true);
      }
      // TODO move free out.
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "free" ||
           sr->getDecl()->getName() == "cudaFree" ||
           sr->getDecl()->getName() == "cudaFreeHost")) {

        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        mlir::Value arg = Visit(sub).getValue(builder);

        if (arg.getType().isa<mlir::LLVM::LLVMPointerType>()) {
          auto callee = EmitCallee(expr->getCallee());
          auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
          mlir::Value args[] = {builder.create<LLVM::BitcastOp>(
              loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), arg)};
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);
        } else {
          builder.create<mlir::memref::DeallocOp>(loc, arg);
        }
        if (sr->getDecl()->getName() == "cudaFree" ||
            sr->getDecl()->getName() == "cudaFreeHost") {
          auto ty = getMLIRType(expr->getType());
          auto op = builder.create<ConstantIntOp>(loc, 0, ty);
          return std::make_pair(ValueCategory(op, /*isReference*/ false), true);
        }
        // TODO remove me when the free is removed.
        return std::make_pair(ValueCategory(), true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMalloc" ||
           sr->getDecl()->getName() == "cudaMallocHost" ||
           sr->getDecl()->getName() == "cudaMallocPitch")) {
        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        {
          auto dst = Visit(sub).getValue(builder);
          if (auto omt = dst.getType().dyn_cast<MemRefType>()) {
            if (auto mt = omt.getElementType().dyn_cast<MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize = getTypeSize(
                  cast<clang::PointerType>(
                      cast<clang::PointerType>(
                          sub->getType()->getUnqualifiedDesugaredType())
                          ->getPointeeType())
                      ->getPointeeType());
              mlir::Value allocSize;
              if (sr->getDecl()->getName() == "cudaMallocPitch") {
                mlir::Value width = Visit(expr->getArg(2)).getValue(builder);
                mlir::Value height = Visit(expr->getArg(3)).getValue(builder);
                // Not changing pitch from provided width here
                // TODO can consider addition alignment considerations
                Visit(expr->getArg(1))
                    .dereference(builder)
                    .store(builder, width);
                allocSize = builder.create<MulIOp>(loc, width, height);
              } else
                allocSize = Visit(expr->getArg(1)).getValue(builder);
              auto idxType = mlir::IndexType::get(builder.getContext());
              mlir::Value args[1] = {builder.create<DivUIOp>(
                  loc, builder.create<IndexCastOp>(loc, idxType, allocSize),
                  elemSize)};
              auto alloc = builder.create<mlir::memref::AllocOp>(
                  loc,
                  (sr->getDecl()->getName() != "cudaMallocHost" && !CudaLower)
                      ? mlir::MemRefType::get(
                            shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(),
                            CodeGenUtils::wrapIntegerMemorySpace(
                                1, mt.getContext()))
                      : mt,
                  args);
              ValueCategory(dst, /*isReference*/ true)
                  .store(builder,
                         builder.create<mlir::memref::CastOp>(loc, mt, alloc));
              auto retTy = getMLIRType(expr->getType());
              return std::make_pair(
                  ValueCategory(builder.create<ConstantIntOp>(loc, 0, retTy),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }

    auto createBlockIdOp = [&](gpu::Dimension str,
                               mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::BlockIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createBlockDimOp = [&](gpu::Dimension str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::BlockDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createThreadIdOp = [&](gpu::Dimension str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::ThreadIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createGridDimOp = [&](gpu::Dimension str,
                               mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::GridDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    if (auto ME = dyn_cast<MemberExpr>(ic->getSubExpr())) {
      auto memberName = ME->getMemberDecl()->getName();

      if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
          if (sr->getDecl()->getName() == "blockIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }
  }
  return std::make_pair(ValueCategory(), false);
}
mlir::Value MLIRScanner::getConstantIndex(int x) {
  if (constants.find(x) != constants.end()) {
    return constants[x];
  }
  mlir::OpBuilder subbuilder(builder.getContext());
  subbuilder.setInsertionPointToStart(entryBlock);
  return constants[x] = subbuilder.create<ConstantIndexOp>(loc, x);
}

ValueCategory MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  auto loc = getMLIRLocation(U->getExprLoc());
  auto sub = Visit(U->getSubExpr());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_LNot: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (auto MT = val.getType().dyn_cast<mlir::MemRefType>()) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(),
                                     MT.getMemorySpaceAsInt()),
          val);
    }

    if (auto LT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, val, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      val = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, val,
          builder.create<ConstantIntOp>(loc, 0, ty));
    }
    auto c1 = builder.create<ConstantIntOp>(loc, 1, val.getType());
    mlir::Value res = builder.create<XOrIOp>(loc, val, c1);

    auto postTy = getMLIRType(U->getType()).cast<mlir::IntegerType>();
    if (postTy.getWidth() > 1)
      res = builder.create<arith::ExtUIOp>(loc, postTy, res);
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    auto c1 = builder.create<ConstantIntOp>(
        loc, APInt::getAllOnesValue(ty.getWidth()).getSExtValue(), ty);
    return ValueCategory(builder.create<XOrIOp>(loc, val, c1),
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(builder);
    return dref;
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(sub.isReference);
    if (sub.val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(sub.val, /*isReference*/ false);
    }

    bool isArray = false;
    Glob.getMLIRType(U->getSubExpr()->getType(), &isArray);
    auto mt = sub.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value res;
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    res = builder.create<memref::CastOp>(loc, mt0, sub.val);
    return ValueCategory(res,
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Plus: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    mlir::Value val = sub.getValue(builder);
    auto ty = val.getType();
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (auto CI = val.getDefiningOp<ConstantFloatOp>()) {
        auto api = CI.getValue().cast<FloatAttr>().getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::FloatAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(builder.create<NegFOp>(loc, val),
                           /*isReference*/ false);
    } else {
      if (auto CI = val.getDefiningOp<ConstantIntOp>()) {
        auto api = CI.getValue().cast<IntegerAttr>().getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::IntegerAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(
          builder.create<SubIOp>(loc,
                                 builder.create<ConstantIntOp>(
                                     loc, 0, ty.cast<mlir::IntegerType>()),
                                 val),
          /*isReference*/ false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(sub.isReference);
    auto prev = sub.getValue(builder);
    auto ty = prev.getType();

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(1));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, 1, ity)}));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);

    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc)
      return sub;
    else
      return ValueCategory(prev, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto ty = getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(builder);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      next = builder.create<SubFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, -1, ity)}));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(-1));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      next = builder.create<SubIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);
    return ValueCategory(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Real:
  case clang::UnaryOperator::Opcode::UO_Imag: {
    int fnum =
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_Real) ? 0 : 1;
    auto lhs_v = sub.val;
    assert(sub.isReference);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      return ValueCategory(builder.create<polygeist::SubIndexOp>(
                               loc, mt0, lhs_v, getConstantIndex(fnum)),
                           /*isReference*/ true);
    } else if (auto PT =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type ET;
      if (auto ST =
              PT.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
        ET = ST.getBody()[fnum];
      } else {
        ET = PT.getElementType()
                 .cast<mlir::LLVM::LLVMArrayType>()
                 .getElementType();
      }
      mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                            builder.create<ConstantIntOp>(loc, fnum, 32)};
      return ValueCategory(
          builder.create<mlir::LLVM::GEPOp>(
              loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
              lhs_v, vec),
          /*isReference*/ true);
    }

    llvm::errs() << "lhs_v: " << lhs_v << "\n";
    U->dump();
    assert(0 && "unhandled real");
  }
  default: {
    U->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

bool hasAffineArith(Operation *op, AffineExpr &expr,
                    mlir::Value &affineForIndVar) {
  // skip IndexCastOp
  if (isa<IndexCastOp>(op))
    return hasAffineArith(op->getOperand(0).getDefiningOp(), expr,
                          affineForIndVar);

  // induction variable are modelled as memref<1xType>
  // %1 = index_cast %induction : index to i32
  // %2 = alloca() : memref<1xi32>
  // store %1, %2[0] : memref<1xi32>
  // ...
  // %5 = load %2[0] : memref<1xf32>
  if (isa<mlir::memref::LoadOp>(op)) {
    auto load = cast<mlir::memref::LoadOp>(op);
    auto loadOperand = load.getOperand(0);
    if (loadOperand.getType().cast<MemRefType>().getShape().size() != 1)
      return false;
    auto maybeAllocaOp = loadOperand.getDefiningOp();
    if (!isa<mlir::memref::AllocaOp>(maybeAllocaOp))
      return false;
    auto allocaUsers = maybeAllocaOp->getUsers();
    if (llvm::none_of(allocaUsers, [](mlir::Operation *op) {
          if (isa<mlir::memref::StoreOp>(op))
            return true;
          return false;
        }))
      return false;
    for (auto user : allocaUsers)
      if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(user)) {
        auto storeOperand = storeOp.getOperand(0);
        auto maybeIndexCast = storeOperand.getDefiningOp();
        if (!isa<IndexCastOp>(maybeIndexCast))
          return false;
        auto indexCastOperand = maybeIndexCast->getOperand(0);
        if (auto blockArg = indexCastOperand.dyn_cast<mlir::BlockArgument>()) {
          if (auto affineForOp = dyn_cast<mlir::AffineForOp>(
                  blockArg.getOwner()->getParentOp()))
            affineForIndVar = affineForOp.getInductionVar();
          else
            return false;
        }
      }
    return true;
  }

  // at this point we expect only AddIOp or MulIOp
  if ((!isa<AddIOp>(op)) && (!isa<MulIOp>(op))) {
    return false;
  }

  // make sure that the current op has at least one constant operand
  // (ConstantIndexOp or ConstantIntOp)
  if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
        return (isa<ConstantIndexOp>(operand.getDefiningOp()) ||
                isa<ConstantIntOp>(operand.getDefiningOp()));
      }))
    return false;

  // build affine expression by adding or multiplying constants.
  // and keep iterating on the non-constant index
  mlir::Value nonCstOperand = nullptr;
  for (auto operand : op->getOperands()) {
    if (auto constantIndexOp =
            dyn_cast<ConstantIndexOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIndexOp.value();
      else
        expr = expr * constantIndexOp.value();
    } else if (auto constantIntOp =
                   dyn_cast<ConstantIntOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIntOp.value();
      else
        expr = expr * constantIntOp.value();
    } else
      nonCstOperand = operand;
  }
  return hasAffineArith(nonCstOperand.getDefiningOp(), expr, affineForIndVar);
}

ValueCategory MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  auto fixInteger = [&](mlir::Value res) {
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    auto postTy = getMLIRType(BO->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (postTy != prevTy) {
      if (signedType) {
        res = builder.create<mlir::arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<mlir::arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  };

  auto lhs = Visit(BO->getLHS());
  if (!lhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->dump();
    BO->getLHS()->dump();
    assert(lhs.val);
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    if (!cond.getType().isa<mlir::IntegerType>()) {
      BO->dump();
      BO->getType()->dump();
      llvm::errs() << "cond: " << cond << "\n";
    }
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, prevTy));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    auto rhs = Visit(BO->getRHS()).getValue(builder);
    assert(rhs != nullptr);
    if (auto LT = rhs.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      rhs = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, rhs, nullptr_llvm);
    }
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, 0, rhs.getType()));
    }
    mlir::Value truearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    mlir::Value falsearray[] = {
        builder.create<ConstantIntOp>(loc, 0, types[0])};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return fixInteger(ifOp.getResult(0));
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, prevTy));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    mlir::Value truearray[] = {builder.create<ConstantIntOp>(loc, 1, types[0])};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    auto rhs = Visit(BO->getRHS()).getValue(builder);
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, 0, rhs.getType()));
    }
    assert(rhs != nullptr);
    mlir::Value falsearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);

    return fixInteger(ifOp.getResult(0));
  }
  default:
    break;
  }
  auto rhs = Visit(BO->getRHS());
  if (!rhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->getRHS()->dump();
    assert(rhs.val);
  }
  // TODO note assumptions made here about unsigned / unordered
  bool signedType = true;
  if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
    if (bit->isUnsignedInteger())
      signedType = false;
    if (bit->isSignedInteger())
      signedType = true;
  }
  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_Shr: {
    auto lhsv = lhs.getValue(builder);
    auto rhsv = rhs.getValue(builder);
    auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
    auto postTy = lhsv.getType().cast<mlir::IntegerType>();
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<mlir::arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<mlir::arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());
    if (signedType)
      return ValueCategory(builder.create<ShRSIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<ShRUIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    auto lhsv = lhs.getValue(builder);
    auto rhsv = rhs.getValue(builder);
    auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
    auto postTy = lhsv.getType().cast<mlir::IntegerType>();
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());
    return ValueCategory(builder.create<ShLIOp>(loc, lhsv, rhsv),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_And: {
    return ValueCategory(builder.create<AndIOp>(loc, lhs.getValue(builder),
                                                rhs.getValue(builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Xor: {
    return ValueCategory(builder.create<XOrIOp>(loc, lhs.getValue(builder),
                                                rhs.getValue(builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
    return ValueCategory(builder.create<OrIOp>(loc, lhs.getValue(builder),
                                               rhs.getValue(builder)),
                         /*isReference*/ false);
  }
    {
      auto lhs_v = lhs.getValue(builder);
      mlir::Value res;
      if (lhs_v.getType().isa<mlir::FloatType>()) {
        res = builder.create<CmpFOp>(loc, CmpFPredicate::UGT, lhs_v,
                                     rhs.getValue(builder));
      } else {
        res = builder.create<CmpIOp>(
            loc, signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt, lhs_v,
            rhs.getValue(builder));
      }
      return fixInteger(res);
    }
  case clang::BinaryOperator::Opcode::BO_GT:
  case clang::BinaryOperator::Opcode::BO_GE:
  case clang::BinaryOperator::Opcode::BO_LT:
  case clang::BinaryOperator::Opcode::BO_LE:
  case clang::BinaryOperator::Opcode::BO_EQ:
  case clang::BinaryOperator::Opcode::BO_NE: {
    signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getLHS()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    CmpFPredicate FPred;
    CmpIPredicate IPred;
    LLVM::ICmpPredicate LPred;
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_GT:
      FPred = CmpFPredicate::UGT;
      IPred = signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt,
      LPred = LLVM::ICmpPredicate::ugt;
      break;
    case clang::BinaryOperator::Opcode::BO_GE:
      FPred = CmpFPredicate::UGE;
      IPred = signedType ? CmpIPredicate::sge : CmpIPredicate::uge,
      LPred = LLVM::ICmpPredicate::uge;
      break;
    case clang::BinaryOperator::Opcode::BO_LT:
      FPred = CmpFPredicate::ULT;
      IPred = signedType ? CmpIPredicate::slt : CmpIPredicate::ult,
      LPred = LLVM::ICmpPredicate::ult;
      break;
    case clang::BinaryOperator::Opcode::BO_LE:
      FPred = CmpFPredicate::ULE;
      IPred = signedType ? CmpIPredicate::sle : CmpIPredicate::ule,
      LPred = LLVM::ICmpPredicate::ule;
      break;
    case clang::BinaryOperator::Opcode::BO_EQ:
      FPred = CmpFPredicate::UEQ;
      IPred = CmpIPredicate::eq;
      LPred = LLVM::ICmpPredicate::eq;
      break;
    case clang::BinaryOperator::Opcode::BO_NE:
      FPred = CmpFPredicate::UNE;
      IPred = CmpIPredicate::ne;
      LPred = LLVM::ICmpPredicate::ne;
      break;
    default:
      llvm_unreachable("Unknown op in binary comparision switch");
    }

    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          lhs_v);
    }
    if (auto mt = rhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          rhs_v);
    }
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<arith::CmpFOp>(loc, FPred, lhs_v, rhs_v);
    } else if (lhs_v.getType().isa<LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>(loc, LPred, lhs_v, rhs_v);
    } else {
      res = builder.create<arith::CmpIOp>(loc, IPred, lhs_v, rhs_v);
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::MulFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else {
      return ValueCategory(
          builder.create<arith::MulIOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::DivFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
      ;
    } else {
      if (signedType)
        return ValueCategory(
            builder.create<arith::DivSIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
      else
        return ValueCategory(
            builder.create<arith::DivUIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::RemFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else {
      if (signedType)
        return ValueCategory(
            builder.create<arith::RemSIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
      else
        return ValueCategory(
            builder.create<arith::RemUIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(builder.create<AddFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      auto ptradd = rhs_v;
      ptradd = castToIndex(loc, ptradd);
      return ValueCategory(
          builder.create<polygeist::SubIndexOp>(loc, mt0, lhs_v, ptradd),
          /*isReference*/ false);
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(
          builder.create<LLVM::GEPOp>(loc, pt, lhs_v,
                                      std::vector<mlir::Value>({rhs_v})),
          /*isReference*/ false);
    } else {
      if (auto lhs_c = lhs_v.getDefiningOp<ConstantIntOp>()) {
        if (auto rhs_c = rhs_v.getDefiningOp<ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(
                  loc, lhs_c.value() + rhs_c.value(), lhs_c.getType()),
              false);
        }
      }
      return ValueCategory(builder.create<AddIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          lhs_v);
    }
    if (auto mt = rhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          rhs_v);
    }
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      assert(rhs_v.getType() == lhs_v.getType());
      return ValueCategory(builder.create<SubFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      if (auto IT = rhs_v.getType().dyn_cast<mlir::IntegerType>()) {
        mlir::Value vals[1] = {builder.create<SubIOp>(
            loc, builder.create<ConstantIntOp>(loc, 0, IT.getWidth()), rhs_v)};
        return ValueCategory(
            builder.create<LLVM::GEPOp>(loc, lhs_v.getType(), lhs_v,
                                        ArrayRef<mlir::Value>(vals)),
            false);
      }
      mlir::Value val =
          builder.create<SubIOp>(loc,
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), lhs_v),
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), rhs_v));
      val = builder.create<DivSIOp>(
          loc, val,
          builder.create<IndexCastOp>(
              loc, val.getType(),
              builder.create<polygeist::TypeSizeOp>(
                  loc, builder.getIndexType(),
                  mlir::TypeAttr::get(pt.getElementType()))));
      return ValueCategory(val, /*isReference*/ false);
    } else {
      return ValueCategory(builder.create<SubIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Assign: {
    assert(lhs.isReference);
    mlir::Value tostore = rhs.getValue(builder);
    mlir::Type subType;
    if (auto PT = lhs.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>())
      subType = PT.getElementType();
    else
      subType = lhs.val.getType().cast<MemRefType>().getElementType();
    if (tostore.getType() != subType) {
      if (auto prevTy = tostore.getType().dyn_cast<mlir::IntegerType>()) {
        if (auto postTy = subType.dyn_cast<mlir::IntegerType>()) {
          bool signedType = true;
          if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
            if (bit->isUnsignedInteger())
              signedType = false;
            if (bit->isSignedInteger())
              signedType = true;
          }

          if (prevTy.getWidth() < postTy.getWidth()) {
            if (signedType) {
              tostore = builder.create<arith::ExtSIOp>(loc, postTy, tostore);
            } else {
              tostore = builder.create<arith::ExtUIOp>(loc, postTy, tostore);
            }
          } else if (prevTy.getWidth() > postTy.getWidth()) {
            tostore = builder.create<arith::TruncIOp>(loc, postTy, tostore);
          }
        }
      }
    }
    lhs.store(builder, tostore);
    return lhs;
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  case clang::BinaryOperator::Opcode::BO_AddAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (auto postTy = prev.getType().dyn_cast<mlir::FloatType>()) {
      mlir::Value rhsV = rhs.getValue(builder);
      auto prevTy = rhsV.getType().cast<mlir::FloatType>();
      if (prevTy == postTy) {
      } else if (prevTy.getWidth() < postTy.getWidth()) {
        rhsV = builder.create<mlir::arith::ExtFOp>(loc, postTy, rhsV);
      } else {
        rhsV = builder.create<mlir::arith::TruncFOp>(loc, postTy, rhsV);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<AddFOp>(loc, prev, rhsV);
    } else if (auto pt =
                   prev.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      result = builder.create<LLVM::GEPOp>(
          loc, pt, prev, std::vector<mlir::Value>({rhs.getValue(builder)}));
    } else if (auto postTy = prev.getType().dyn_cast<mlir::IntegerType>()) {
      mlir::Value rhsV = rhs.getValue(builder);
      auto prevTy = rhsV.getType().cast<mlir::IntegerType>();
      if (prevTy == postTy) {
      } else if (prevTy.getWidth() < postTy.getWidth()) {
        if (signedType) {
          rhsV = builder.create<arith::ExtSIOp>(loc, postTy, rhsV);
        } else {
          rhsV = builder.create<arith::ExtUIOp>(loc, postTy, rhsV);
        }
      } else {
        rhsV = builder.create<arith::TruncIOp>(loc, postTy, rhsV);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<AddIOp>(loc, prev, rhsV);
    } else if (auto postTy = prev.getType().dyn_cast<mlir::MemRefType>()) {
      mlir::Value rhsV = rhs.getValue(builder);
      auto shape = std::vector<int64_t>(postTy.getShape());
      shape[0] = -1;
      postTy = mlir::MemRefType::get(shape, postTy.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     postTy.getMemorySpace());
      auto ptradd = rhsV;
      ptradd = castToIndex(loc, ptradd);
      result = builder.create<polygeist::SubIndexOp>(loc, postTy, prev, ptradd);
    } else {
      assert(false && "Unsupported add assign type");
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_SubAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      auto right = rhs.getValue(builder);
      if (right.getType() != prev.getType()) {
        auto prevTy = right.getType().cast<mlir::FloatType>();
        auto postTy = getMLIRType(BO->getType()).cast<mlir::FloatType>();

        if (prevTy.getWidth() < postTy.getWidth()) {
          right = builder.create<arith::ExtFOp>(loc, postTy, right);
        } else {
          right = builder.create<arith::TruncFOp>(loc, postTy, right);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<SubFOp>(loc, prev, right);
    } else {
      result = builder.create<SubIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      auto right = rhs.getValue(builder);
      if (right.getType() != prev.getType()) {
        auto prevTy = right.getType().cast<mlir::FloatType>();
        auto postTy = getMLIRType(BO->getType()).cast<mlir::FloatType>();

        if (prevTy.getWidth() < postTy.getWidth()) {
          right = builder.create<arith::ExtFOp>(loc, postTy, right);
        } else {
          right = builder.create<arith::TruncFOp>(loc, postTy, right);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<MulFOp>(loc, prev, right);
    } else {
      result = builder.create<MulIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_DivAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      mlir::Value val = rhs.getValue(builder);
      auto prevTy = val.getType().cast<mlir::FloatType>();
      auto postTy = prev.getType().cast<mlir::FloatType>();

      if (prevTy.getWidth() < postTy.getWidth()) {
        val = builder.create<arith::ExtFOp>(loc, postTy, val);
      } else if (prevTy.getWidth() > postTy.getWidth()) {
        val = builder.create<arith::TruncFOp>(loc, postTy, val);
      }
      result = builder.create<arith::DivFOp>(loc, prev, val);
    } else {
      if (signedType)
        result =
            builder.create<arith::DivSIOp>(loc, prev, rhs.getValue(builder));
      else
        result =
            builder.create<arith::DivUIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (signedType)
      result = builder.create<ShRSIOp>(loc, prev, rhs.getValue(builder));
    else
      result = builder.create<ShRUIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShlAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<ShLIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_RemAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<RemFOp>(loc, prev, rhs.getValue(builder));
    } else {
      if (signedType)
        result = builder.create<RemSIOp>(loc, prev, rhs.getValue(builder));
      else
        result = builder.create<RemUIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_AndAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<AndIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_OrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<OrIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_XorAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<XOrIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }

  default: {
    BO->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::CommonFieldLookup(clang::QualType CT,
                                             const FieldDecl *FD,
                                             mlir::Value val, bool isLValue) {
  assert(FD && "Attempting to lookup field of nullptr");
  auto rd = FD->getParent();

  auto ST = cast<llvm::StructType>(getLLVMType(CT));

  size_t fnum = 0;

  auto CXRD = dyn_cast<CXXRecordDecl>(rd);

  if (CodeGenUtils::isLLVMStructABI(rd, ST)) {
    auto &layout = Glob.getCGM().getTypes().getCGRecordLayout(rd);
    fnum = layout.getLLVMFieldNo(FD);
  } else {
    fnum = 0;
    if (CXRD)
      fnum += CXRD->getDefinition()->getNumBases();
    for (auto field : rd->fields()) {
      if (field == FD) {
        break;
      }
      fnum++;
    }
  }

  if (auto PT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    mlir::Value vec[] = {builder.create<ConstantIntOp>(loc, 0, 32),
                         builder.create<ConstantIntOp>(loc, fnum, 32)};
    if (!PT.getElementType()
             .isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>()) {
      llvm::errs() << "function: " << function << "\n";
      // rd->dump();
      FD->dump();
      FD->getType()->dump();
      llvm::errs() << " val: " << val << " - pt: " << PT << " fn: " << fnum
                   << " ST: " << *ST << "\n";
    }
    mlir::Type ET =
        mlir::TypeSwitch<mlir::Type, mlir::Type>(PT.getElementType())
            .Case<mlir::LLVM::LLVMStructType>(
                [fnum](mlir::LLVM::LLVMStructType ST) {
                  return ST.getBody()[fnum];
                })
            .Case<mlir::LLVM::LLVMArrayType>([](mlir::LLVM::LLVMArrayType AT) {
              return AT.getElementType();
            })
            .Case<MemRefType>([](MemRefType MT) { return MT.getElementType(); })
            .Default([](mlir::Type T) {
              llvm_unreachable("not implemented");
              return T;
            });

    mlir::Value commonGEP = builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), val,
        vec);
    if (rd->isUnion()) {
      auto subType =
          Glob.getTypeTranslator().translateType(getLLVMType(FD->getType()));
      commonGEP = builder.create<mlir::LLVM::BitcastOp>(
          loc, mlir::LLVM::LLVMPointerType::get(subType, PT.getAddressSpace()),
          commonGEP);
    }
    if (isLValue)
      commonGEP =
          ValueCategory(commonGEP, /*isReference*/ true).getValue(builder);
    return ValueCategory(commonGEP, /*isReference*/ true);
  }
  auto mt = val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }

  // JLE_QUEL::THOUGHTS
  // This redundancy is here because we might, at some point, create
  // an equivalent GEP or SubIndexOp operation for each sycl types or otherwise
  // clean the redundancy
  mlir::Value Result;
  if (auto ST = mt.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
    assert(fnum < ST.getBody().size() && "ERROR");

    const auto ElementType = ST.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto AT =
                 mt.getElementType().dyn_cast<mlir::sycl::AccessorType>()) {
    assert(fnum < AT.getBody().size() && "ERROR");

    const auto ElementType = AT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto AT = mt.getElementType()
                           .dyn_cast<mlir::sycl::AccessorImplDeviceType>()) {
    assert(fnum < AT.getBody().size() && "ERROR");

    const auto ElementType = AT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto AT = mt.getElementType().dyn_cast<mlir::sycl::ArrayType>()) {
    assert(fnum < AT.getBody().size() && "ERROR");
    const auto elemType = AT.getBody()[fnum].cast<MemRefType>();
    const auto ResultType =
        mlir::MemRefType::get(elemType.getShape(), elemType.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto IT = mt.getElementType().dyn_cast<mlir::sycl::IDType>()) {
    llvm_unreachable("not implemented");
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::RangeType>()) {
    llvm_unreachable("not implemented");
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::ItemType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT =
                 mt.getElementType().dyn_cast<mlir::sycl::ItemBaseType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::NdItemType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::GroupType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else {
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    shape[0] = -1;
    auto mt1 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, mt0, val,
                                                   getConstantIndex(0));
    Result = builder.create<polygeist::SubIndexOp>(loc, mt1, Result,
                                                   getConstantIndex(fnum));
  }

  if (isLValue) {
    Result = ValueCategory(Result, /*isReference*/ true).getValue(builder);
  }

  return ValueCategory(Result, /*isReference*/ true);
}

static bool isSyclIDorRangetoArray(mlir::Type &nt, mlir::Value &value) {
  mlir::Type elemTy = value.getType().dyn_cast<MemRefType>().getElementType();
  return ((elemTy.isa<sycl::IDType>() || elemTy.isa<sycl::RangeType>()) &&
          nt.dyn_cast<MemRefType>().getElementType().isa<sycl::ArrayType>());
}

mlir::Value MLIRScanner::GetAddressOfDerivedClass(
    mlir::Value value, const CXXRecordDecl *DerivedClass,
    CastExpr::path_const_iterator Start, CastExpr::path_const_iterator End) {
  const ASTContext &Context = Glob.getCGM().getContext();

  SmallVector<const CXXRecordDecl *> ToBase = {DerivedClass};
  SmallVector<const CXXBaseSpecifier *> Bases;
  for (auto I = Start; I != End; I++) {
    const CXXBaseSpecifier *Base = *I;
    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    ToBase.push_back(BaseDecl);
    Bases.push_back(Base);
  }

  for (int i = ToBase.size() - 1; i > 0; i--) {
    const CXXBaseSpecifier *Base = Bases[i - 1];

    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    const auto *RD = ToBase[i - 1];
    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Add the offset.

    mlir::Type nt = getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(Base->getType()));

    mlir::Value Offset = nullptr;
    if (CodeGenUtils::isLLVMStructABI(RD, /*ST*/ nullptr)) {
      Offset = builder.create<arith::ConstantIntOp>(
          loc, -(ssize_t)Layout.getBaseClassOffset(BaseDecl).getQuantity(), 32);
    } else {
      Offset = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            Base->getType()->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        bool subType = false;
        mlir::Type nt = Glob.getMLIRType(f.getType(), &subType, false);
        Offset = builder.create<arith::SubIOp>(
            loc, Offset,
            builder.create<IndexCastOp>(
                loc, Offset.getType(),
                builder.create<polygeist::TypeSizeOp>(
                    loc, builder.getIndexType(), mlir::TypeAttr::get(nt))));
      }
      assert(found);
    }

    mlir::Value ptr = value;
    if (auto PT = ptr.getType().dyn_cast<LLVM::LLVMPointerType>())
      ptr = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(), PT.getAddressSpace()),
          ptr);
    else
      ptr = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(), PT.getAddressSpace()),
          ptr);

    mlir::Value idx[] = {Offset};
    ptr = builder.create<LLVM::GEPOp>(loc, ptr.getType(), ptr, idx);

    if (auto PT = nt.dyn_cast<mlir::LLVM::LLVMPointerType>())
      value = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              PT.getElementType(),
              ptr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          ptr);
    else
      value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, ptr);
  }

  return value;
}

mlir::Value MLIRScanner::GetAddressOfBaseClass(
    mlir::Value value, const CXXRecordDecl *DerivedClass,
    ArrayRef<const clang::Type *> BaseTypes, ArrayRef<bool> BaseVirtuals) {
  const CXXRecordDecl *RD = DerivedClass;

  for (auto tup : llvm::zip(BaseTypes, BaseVirtuals)) {

    auto BaseType = std::get<0>(tup);

    const auto *BaseDecl =
        cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());
    // Add the offset.

    mlir::Type nt =
        getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
            QualType(BaseType, 0)));

    size_t fnum;
    bool subIndex = true;

    if (CodeGenUtils::isLLVMStructABI(RD, /*ST*/ nullptr)) {
      auto &layout = Glob.getCGM().getTypes().getCGRecordLayout(RD);
      if (std::get<1>(tup))
        fnum = layout.getVirtualBaseIndex(BaseDecl);
      else {
        if (!layout.hasNonVirtualBaseLLVMField(BaseDecl)) {
          subIndex = false;
        } else {
          fnum = layout.getNonVirtualBaseLLVMFieldNo(BaseDecl);
        }
      }
    } else {
      assert(!std::get<1>(tup) && "Should not see virtual bases here!");
      fnum = 0;
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            BaseType->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        fnum++;
      }
      assert(found);
    }

    if (subIndex) {
      if (auto mt = value.getType().dyn_cast<MemRefType>()) {
        auto shape = std::vector<int64_t>(mt.getShape());
        // We do not remove dimensions for an id->array or range->array, because
        // the later cast will be incompatible due to dimension mismatch.
        if (!isSyclIDorRangetoArray(nt, value))
          shape.erase(shape.begin());
        auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                         MemRefLayoutAttrInterface(),
                                         mt.getMemorySpace());
        value = builder.create<polygeist::SubIndexOp>(loc, mt0, value,
                                                      getConstantIndex(fnum));
      } else {
        mlir::Value idx[] = {
            builder.create<arith::ConstantIntOp>(loc, 0, 32),
            builder.create<arith::ConstantIntOp>(loc, fnum, 32)};
        auto PT = value.getType().cast<LLVM::LLVMPointerType>();
        mlir::Type ET =
            mlir::TypeSwitch<mlir::Type, mlir::Type>(PT.getElementType())
                .Case<mlir::LLVM::LLVMStructType>(
                    [fnum](mlir::LLVM::LLVMStructType ST) {
                      return ST.getBody()[fnum];
                    })
                .Case<mlir::LLVM::LLVMArrayType>(
                    [](mlir::LLVM::LLVMArrayType AT) {
                      return AT.getElementType();
                    })
                .Case<mlir::sycl::AccessorType>(
                    [fnum](mlir::sycl::AccessorType AT) {
                      return AT.getBody()[fnum];
                    })
                .Default([](mlir::Type T) {
                  llvm_unreachable("not implemented");
                  return T;
                });

        value = builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), value,
            idx);
      }
    }

    auto pt = nt.dyn_cast<mlir::LLVM::LLVMPointerType>();
    if (auto opt = value.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      if (!pt) {
        value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, value);
      } else {
        if (value.getType() != nt)
          value = builder.create<mlir::LLVM::BitcastOp>(loc, pt, value);
      }
    } else {
      assert(value.getType().isa<MemRefType>() &&
             "Expecting value to have MemRefType");
      if (pt) {
        assert(
            value.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                pt.getAddressSpace() &&
            "The type of 'value' does not have the same memory space as 'pt'");
        value = builder.create<polygeist::Memref2PointerOp>(loc, pt, value);
      } else {
        if (value.getType() != nt) {
          if (isSyclIDorRangetoArray(nt, value))
            value = builder.create<sycl::SYCLCastOp>(loc, nt, value);
          else
            value = builder.create<memref::CastOp>(loc, nt, value);
        }
      }
    }

    RD = BaseDecl;
  }

  return value;
}

/******************************************************************************/
/*                             MLIRASTConsumer                                */
/******************************************************************************/

mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateMallocFunction() {
  std::string name = "malloc";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {mlir::IntegerType::get(ctx, 64)};
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8)), types, false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}
mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateFreeFunction() {
  std::string name = "free";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {
      LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8))};
  auto llvmFnType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types, false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}

mlir::LLVM::LLVMFuncOp
MLIRASTConsumer::GetOrCreateLLVMFunction(const FunctionDecl *FD) {
  std::string name = MLIRScanner::getMangledFuncName(*FD, CGM);

  if (name != "malloc" && name != "free")
    name = (PrefixABI + name);

  if (llvmFunctions.find(name) != llvmFunctions.end())
    return llvmFunctions[name];

  std::vector<mlir::Type> types;
  if (auto CC = dyn_cast<CXXMethodDecl>(FD)) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(CC->getThisType()))));
  }
  for (auto parm : FD->parameters()) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(parm->getOriginalType()))));
  }

  auto rt =
      typeTranslator.translateType(anonymize(getLLVMType(FD->getReturnType())));
  auto llvmFnType = LLVM::LLVMFunctionType::get(rt, types,
                                                /*isVarArg=*/FD->isVariadic());
  // Insert the function into the body of the parent module.
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType,
             getMLIRLinkage(CGM.getFunctionLinkage(FD)));
}

mlir::LLVM::GlobalOp
MLIRASTConsumer::GetOrCreateLLVMGlobal(const ValueDecl *FD,
                                       std::string prefix) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  if (llvmGlobals.find(name) != llvmGlobals.end()) {
    return llvmGlobals[name];
  }

  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  auto linkage = CGM.getLLVMLinkageVarDefinition(VD, /*isConstant*/ false);
  LLVM::Linkage lnk = getMLIRLinkage(linkage);

  auto rt = getMLIRType(FD->getType());

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto glob = builder.create<LLVM::GlobalOp>(
      module->getLoc(), rt, /*constant*/ false, lnk, name, mlir::Attribute());

  if (VD->getInit() ||
      VD->isThisDeclarationADefinition() == VarDecl::Definition ||
      VD->isThisDeclarationADefinition() == VarDecl::TentativeDefinition) {
    Block *blk = new Block();
    builder.setInsertionPointToStart(blk);
    mlir::Value res;
    if (auto init = VD->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      ms.setEntryAndAllocBlock(blk);
      res = ms.Visit(const_cast<Expr *>(init)).getValue(builder);
    } else {
      res = builder.create<LLVM::UndefOp>(module->getLoc(), rt);
    }
    bool legal = true;
    for (Operation &op : *blk) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect()) {
        legal = false;
        break;
      }
    }
    if (legal) {
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({res}));
      glob.getInitializerRegion().push_back(blk);
    } else {
      Block *blk2 = new Block();
      builder.setInsertionPointToEnd(blk2);
      mlir::Value nres = builder.create<LLVM::UndefOp>(module->getLoc(), rt);
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({nres}));
      glob.getInitializerRegion().push_back(blk2);

      builder.setInsertionPointToStart(module->getBody());
      auto funcName = name + "@init";
      LLVM::GlobalCtorsOp ctors = nullptr;
      for (auto &op : *module->getBody()) {
        if (auto c = dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
          ctors = c;
        }
      }
      SmallVector<mlir::Attribute> funcs;
      funcs.push_back(FlatSymbolRefAttr::get(module->getContext(), funcName));
      SmallVector<mlir::Attribute> idxs;
      idxs.push_back(builder.getI32IntegerAttr(0));
      if (ctors) {
        for (auto f : ctors.getCtors())
          funcs.push_back(f);
        for (auto v : ctors.getPriorities())
          idxs.push_back(v);
        ctors->erase();
      }

      builder.create<LLVM::GlobalCtorsOp>(module->getLoc(),
                                          builder.getArrayAttr(funcs),
                                          builder.getArrayAttr(idxs));

      auto llvmFnType = LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMVoidType::get(module->getContext()),
          ArrayRef<mlir::Type>(), false);

      auto func = builder.create<LLVM::LLVMFuncOp>(
          module->getLoc(), funcName, llvmFnType, LLVM::Linkage::Private);
      func.getRegion().push_back(blk);
      builder.setInsertionPointToEnd(blk);
      builder.create<LLVM::StoreOp>(
          module->getLoc(), res,
          builder.create<LLVM::AddressOfOp>(module->getLoc(), glob));
      builder.create<LLVM::ReturnOp>(module->getLoc(), ArrayRef<mlir::Value>());
    }
  }
  if (lnk == LLVM::Linkage::Private || lnk == LLVM::Linkage::Internal) {
    SymbolTable::setSymbolVisibility(glob,
                                     mlir::SymbolTable::Visibility::Private);
  }
  return llvmGlobals[name] = glob;
}

std::pair<mlir::memref::GlobalOp, bool>
MLIRASTConsumer::GetOrCreateGlobal(const ValueDecl *FD, std::string prefix,
                                   bool tryInit, FunctionContext funcContext) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  if (globals.find(name) != globals.end()) {
    return globals[name];
  }

  auto rt = getMLIRType(FD->getType());
  unsigned memspace = 0;
  bool isArray = isa<clang::ArrayType>(FD->getType());
  bool isExtVectorType =
      isa<clang::ExtVectorType>(FD->getType()->getUnqualifiedDesugaredType());

  mlir::MemRefType mr;
  if (!isArray && !isExtVectorType) {
    mr = mlir::MemRefType::get(1, rt, {}, memspace);
  } else {
    auto mt = rt.cast<mlir::MemRefType>();
    mr = mlir::MemRefType::get(
        mt.getShape(), mt.getElementType(), MemRefLayoutAttrInterface(),
        CodeGenUtils::wrapIntegerMemorySpace(memspace, mt.getContext()));
  }

  mlir::SymbolTable::Visibility lnk;
  mlir::Attribute initial_value;

  mlir::OpBuilder builder(module->getContext());
  if (funcContext == FunctionContext::SYCLDevice)
    builder.setInsertionPointToStart(
        &(getDeviceModule(*module).body().front()));
  else
    builder.setInsertionPointToStart(module->getBody());

  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  if (VD->isThisDeclarationADefinition() == VarDecl::Definition) {
    initial_value = builder.getUnitAttr();
  } else if (VD->isThisDeclarationADefinition() ==
             VarDecl::TentativeDefinition) {
    initial_value = builder.getUnitAttr();
  }

  switch (CGM.getLLVMLinkageVarDefinition(VD,
                                          /*isConstant*/ false)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = mlir::SymbolTable::Visibility::Private;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  }
  auto globalOp = builder.create<mlir::memref::GlobalOp>(
      module->getLoc(), builder.getStringAttr(name),
      /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
      initial_value, mlir::UnitAttr(), /*alignment*/ nullptr);
  SymbolTable::setSymbolVisibility(globalOp, lnk);

  globals[name] = std::make_pair(globalOp, isArray);

  if (tryInit)
    if (auto init = VD->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      mlir::Block *B = new Block();
      // In case of device function, we will put the block in the forefront of
      // the GPU module, else the block will go at the forefront of the main
      // module.
      if (funcContext == FunctionContext::SYCLDevice) {
        B->moveBefore(&(getDeviceModule(*module).body().front()));
        ms.getBuilder().setInsertionPointToStart(B);
      } else
        ms.setEntryAndAllocBlock(B);
      OpBuilder builder(module->getContext());
      builder.setInsertionPointToEnd(B);
      auto op = builder.create<memref::AllocaOp>(module->getLoc(), mr);

      bool initialized = false;
      if (isa<InitListExpr>(init)) {
        if (auto A = ms.InitializeValueByInitListExpr(
                op, const_cast<clang::Expr *>(init))) {
          initialized = true;
          initial_value = A;
        }
      } else {
        auto VC = ms.Visit(const_cast<clang::Expr *>(init));
        if (!VC.isReference) {
          if (auto cop = VC.val.getDefiningOp<arith::ConstantOp>()) {
            initial_value = cop.getValue();
            initial_value = SplatElementsAttr::get(
                RankedTensorType::get(mr.getShape(), mr.getElementType()),
                initial_value);
            initialized = true;
          }
        }
      }

      if (!initialized) {
        FD->dump();
        init->dump();
        llvm::errs() << " warning not initializing global: " << name << "\n";
      } else {
        globalOp.initial_valueAttr(initial_value);
      }
      delete B;
    }

  return globals[name];
}

mlir::Value MLIRASTConsumer::GetOrCreateGlobalLLVMString(
    mlir::Location loc, mlir::OpBuilder &builder, StringRef value) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module->getBody());
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
    llvmStringGlobals[value.str()] = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
        "str" + std::to_string(llvmStringGlobals.size()),
        builder.getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  return globalPtr;
}

mlir::FunctionOpInterface
MLIRASTConsumer::GetOrCreateMLIRFunction(FunctionToEmit &FTE, bool IsThunk,
                                         bool ShouldEmit, bool getDeviceStub) {
  assert(FTE.getDecl().getTemplatedKind() !=
             FunctionDecl::TemplatedKind::TK_FunctionTemplate &&
         FTE.getDecl().getTemplatedKind() !=
             FunctionDecl::TemplatedKind::
                 TK_DependentFunctionTemplateSpecialization &&
         "Unexpected template kind");

  const clang::FunctionDecl &FD = FTE.getDecl();
  const std::string mangledName =
      (getDeviceStub)
          ? PrefixABI +
                CGM.getMangledName(GlobalDecl(&FD, KernelReferenceKind::Kernel))
                    .str()
          : PrefixABI + MLIRScanner::getMangledFuncName(FD, CGM);
  assert(mangledName != "free");

  // Early exit if the function has already been generated.
  if (Optional<FunctionOpInterface> optFunction =
          getMLIRFunction(mangledName, FTE.getContext()))
    return *optFunction;

  // Create the MLIR function and set its various attributes.
  FunctionOpInterface function =
      createMLIRFunction(FTE, mangledName, IsThunk, ShouldEmit);
  checkFunctionParent(function, FTE.getContext(), module);

  // Decide whether the MLIR function should be emitted.
  const FunctionDecl *Def = nullptr;
  if (!FD.isDefined(Def, /*checkforfriend*/ true))
    Def = &FD;

  if (Def->isThisDeclarationADefinition()) {
    assert(Def->getTemplatedKind() !=
               FunctionDecl::TemplatedKind::TK_FunctionTemplate &&
           Def->getTemplatedKind() !=
               FunctionDecl::TemplatedKind::
                   TK_DependentFunctionTemplateSpecialization);
    if (ShouldEmit) {
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << Def->getNameAsString() << " to functionsToEmit\n");
      functionsToEmit.emplace_back(*Def, FTE.getContext());
    }
  } else if (ShouldEmit) {
    emitIfFound.insert(mangledName);
  }

  return function;
}

void MLIRASTConsumer::run() {
  while (functionsToEmit.size()) {
    FunctionToEmit &FTE = functionsToEmit.front();
    functionsToEmit.pop_front();

    const FunctionDecl &FD = FTE.getDecl();

    assert(FD.getBody());
    assert(FD.getTemplatedKind() != FunctionDecl::TK_FunctionTemplate);
    assert(FD.getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);

    std::string mangledName = MLIRScanner::getMangledFuncName(FD, CGM);
    const std::pair<FunctionContext, std::string> doneKey(FTE.getContext(),
                                                          mangledName);
    if (done.count(doneKey))
      continue;

    LLVM_DEBUG({
      StringRef funcKind =
          (FD.hasAttr<clang::SYCLKernelAttr>()   ? "SYCL KERNEL"
           : FD.hasAttr<clang::SYCLDeviceAttr>() ? "SYCL DEVICE"
                                                 : "");
      llvm::dbgs() << "\n-- " << funcKind << " FUNCTION (" << FTE.getContext()
                   << " context) BEING EMITTED: " << FD.getNameAsString()
                   << " --\n\n";
    });

    done.insert(doneKey);
    MLIRScanner ms(*this, module, LTInfo);
    FunctionOpInterface function = GetOrCreateMLIRFunction(
        FTE, false /* IsThunk */, true /* ShouldEmit */);
    ms.init(function, FTE);

    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      function.dump();
      llvm::dbgs() << "\n";

      if (functionsToEmit.size()) {
        llvm::dbgs() << "-- FUNCTION(S) LEFT TO BE EMITTED --\n";

        for (const auto &FTE : functionsToEmit) {
          const clang::FunctionDecl &FD = FTE.getDecl();
          llvm::dbgs() << "  [+] " << FD.getNameAsString() << "(";
          for (unsigned int index = 0; index < FD.getNumParams(); index += 1) {
            printf("%s",
                   FD.getParamDecl(index)->getType().getAsString().c_str());
            if (index + 1 != FD.getNumParams())
              llvm::dbgs() << ", ";
          }
          llvm::dbgs() << ")\n";
        }
        llvm::dbgs() << "\n";
      }
    });
  }
}

void MLIRASTConsumer::HandleDeclContext(DeclContext *DC) {

  for (auto D : DC->decls()) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    const FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(D);
    if (!fd)
      continue;
    if (!fd->doesThisDeclarationHaveABody() &&
        !fd->doesDeclarationForceExternallyVisibleDefinition())
      continue;
    if (!fd->hasBody())
      continue;
    if (fd->isTemplated())
      continue;

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name = MLIRScanner::getMangledFuncName(*fd, CGM);

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt"))
      continue;
    if (StringRef(name).startswith("_ZSt"))
      continue;
    if (StringRef(name).startswith("_ZNSt"))
      continue;
    if (StringRef(name).startswith("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name)) {
      FunctionToEmit FTE(*fd);
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << fd->getNameAsString() << " to functionsToEmit\n");
      functionsToEmit.push_back(FTE);
    }
  }
}

bool MLIRASTConsumer::HandleTopLevelDecl(DeclGroupRef dg) {
  DeclGroupRef::iterator it;

  if (error)
    return true;

  for (it = dg.begin(); it != dg.end(); ++it) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    const FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
    if (!fd)
      continue;
    if (!fd->doesThisDeclarationHaveABody() &&
        !fd->doesDeclarationForceExternallyVisibleDefinition())
      continue;
    if (!fd->hasBody())
      continue;
    if (fd->isTemplated())
      continue;

    //  if (fd->getIdentifier())
    //    llvm::errs() << "Func name: " << fd->getName() << "\n";
    //  llvm::errs() << "Func Body && Loc " << "\n";
    //  fd->getBody()->dump();
    //  fd->getLocation().dump(SM);

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name = MLIRScanner::getMangledFuncName(*fd, CGM);

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt"))
      continue;
    if (StringRef(name).startswith("_ZSt"))
      continue;
    if (StringRef(name).startswith("_ZNSt"))
      continue;
    if (StringRef(name).startswith("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name) || fd->hasAttr<OpenCLKernelAttr>() ||
        fd->hasAttr<SYCLDeviceAttr>()) {
      FunctionToEmit FTE(*fd);
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << fd->getNameAsString() << " to functionsToEmit\n");
      functionsToEmit.push_back(FTE);
    }
  }

  return true;
}

// Wait until Sema has instantiated all the relevant code
// before running codegen on the selected functions.
void MLIRASTConsumer::HandleTranslationUnit(ASTContext &C) { run(); }

mlir::Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation loc) {
  auto spellingLoc = SM.getSpellingLoc(loc);
  auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
  auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
  auto fileId = SM.getFilename(spellingLoc);

  auto ctx = module->getContext();
  return FileLineColLoc::get(ctx, fileId, lineNumber, colNumber);
}

llvm::GlobalValue::LinkageTypes
MLIRASTConsumer::getLLVMLinkageType(const clang::FunctionDecl &FD,
                                    bool shouldEmit) {
  if (!FD.hasBody() || !shouldEmit)
    return llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  if (auto CC = dyn_cast<CXXConstructorDecl>(&FD))
    return CGM.getFunctionLinkage(GlobalDecl(CC, CXXCtorType::Ctor_Complete));
  if (auto CC = dyn_cast<CXXDestructorDecl>(&FD))
    return CGM.getFunctionLinkage(GlobalDecl(CC, CXXDtorType::Dtor_Complete));

  return CGM.getFunctionLinkage(&FD);
}

mlir::LLVM::Linkage
MLIRASTConsumer::getMLIRLinkage(llvm::GlobalValue::LinkageTypes LV) {
  switch (LV) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    return LLVM::Linkage::Internal;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    return LLVM::Linkage::External;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    return LLVM::Linkage::AvailableExternally;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    return LLVM::Linkage::Linkonce;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    return LLVM::Linkage::Weak;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    return LLVM::Linkage::WeakODR;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    return LLVM::Linkage::Common;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    return LLVM::Linkage::Appending;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    return LLVM::Linkage::ExternWeak;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    return LLVM::Linkage::LinkonceODR;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    return LLVM::Linkage::Private;
  }

  llvm_unreachable("Unexpected linkage");
}

void MLIRASTConsumer::createMLIRParameterDescriptors(
    const clang::FunctionDecl &FD,
    SmallVectorImpl<CodeGenUtils::ParmDesc> &parmDescriptors) {
  assert(parmDescriptors.empty() && "Expecting 'parmDescriptors' to be empty");

  bool isMethodDecl = isa<CXXMethodDecl>(FD);
  bool isMethodInstance = isMethodDecl && cast<CXXMethodDecl>(FD).isInstance();

  // Handle the this pointer of a method instance.
  if (isMethodInstance) {
    auto &CC = cast<CXXMethodDecl>(FD);
    QualType thisType = CC.getThisType();
    auto mlirThisType = getMLIRType(thisType);

    if (auto mt = mlirThisType.dyn_cast<MemRefType>())
      mlirThisType = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());

    LLVM_DEBUG(if (!mlirThisType.isa<LLVM::LLVMPointerType, MemRefType>()) {
      bool isArray = false;
      getMLIRType(CC.getThisObjectType(), &isArray);

      FD.dump();
      thisType->dump();
      llvm::dbgs() << " mlirThisType: " << mlirThisType
                   << " isArray: " << (int)isArray
                   << " LLTy: " << *getLLVMType(thisType) << "\n";
    });
    assert((mlirThisType.isa<LLVM::LLVMPointerType, MemRefType>()) &&
           "Unexpected type");

    parmDescriptors.push_back(mlirThisType);
  }

  const bool isKernel = FD.hasAttr<SYCLKernelAttr>();
  mlir::MLIRContext *ctx = module->getContext();

  // Handle the remaining parameters.
  for (const ParmVarDecl *parm : FD.parameters()) {
    QualType parmType = parm->getType();
    mlirclang::AttrBuilder attrBuilder(*ctx);

    // SPIR ABI compliance: aggregates need to be passed indirectly as pointers
    // (in MLIR, we pass it as MemRefs with unknown size).
    // Note: indirect arguments are always on the stack (i.e. using alloca addr
    // space).
    if (isKernel && CodeGenUtils::isAggregateTypeForABI(parmType)) {
      auto mt = mlir::MemRefType::get(-1, getMLIRType(parmType), {},
                                      CGM.getDataLayout().getAllocaAddrSpace());
      attrBuilder.addAttribute(llvm::Attribute::AttrKind::ByVal);
      attrBuilder.addAttribute(
          llvm::Attribute::AttrKind::Alignment,
          CGM.getContext().getTypeAlignInChars(parmType).getQuantity());

      const llvm::DataLayout &DL = CGM.getDataLayout();
      if (CGM.getCodeGenOpts().EnableNoundefAttrs &&
          CodeGenUtils::determineNoUndef(parmType, CGM.getTypes(), DL,
                                         CodeGen::ABIArgInfo::Indirect)) {
        attrBuilder.addAttribute(llvm::Attribute::AttrKind::NoUndef);
      }

      CodeGenUtils::ParmDesc parmDesc(mt, attrBuilder.getAttrs());
      parmDescriptors.push_back(parmDesc);

      continue;
    }

    bool ArrayStruct = false;
    auto mlirType = getMLIRType(parmType, &ArrayStruct);
    if (ArrayStruct)
      mlirType = getMLIRType(CGM.getContext().getLValueReferenceType(parmType));

#if 0
    llvm::dbgs() << __LINE__ << "\n";
    llvm::dbgs() << "parmType: ";
    parmType.dump();
    llvm::dbgs() << "\n";
    llvm::dbgs() << "mlirType: " << mlirType << "\n";
#endif

    parmDescriptors.push_back(mlirType);
  }

  bool isArrayReturn = false;
  getMLIRType(FD.getReturnType(), &isArrayReturn);

  if (isArrayReturn) {
    auto mt =
        getMLIRType(CGM.getContext().getLValueReferenceType(FD.getReturnType()))
            .cast<MemRefType>();
    assert(mt.getShape().size() == 2);
    parmDescriptors.push_back(mt);
  }
}

namespace {

/// Encapsulates information about the way function arguments from
/// CGFunctionInfo should be passed to actual LLVM IR function.
class ClangToLLVMArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned InallocaArgNo;
  unsigned SRetArgNo;
  unsigned TotalIRArgs;

  /// Arguments of LLVM IR function corresponding to single Clang argument.
  struct IRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to IR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

    IRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  SmallVector<IRArgs, 8> ArgInfo;

public:
  ClangToLLVMArgMapping(const ASTContext &Context,
                        const CodeGen::CGFunctionInfo &FI,
                        bool OnlyRequiredArgs = false)
      : InallocaArgNo(InvalidIndex), SRetArgNo(InvalidIndex), TotalIRArgs(0),
        ArgInfo(OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size()) {
    construct(Context, FI, OnlyRequiredArgs);
  }

  bool hasInallocaArg() const { return InallocaArgNo != InvalidIndex; }
  unsigned getInallocaArgNo() const {
    assert(hasInallocaArg());
    return InallocaArgNo;
  }

  bool hasSRetArg() const { return SRetArgNo != InvalidIndex; }
  unsigned getSRetArgNo() const {
    assert(hasSRetArg());
    return SRetArgNo;
  }

  unsigned totalIRArgs() const { return TotalIRArgs; }

  bool hasPaddingArg(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return ArgInfo[ArgNo].PaddingArgIndex != InvalidIndex;
  }
  unsigned getPaddingArgNo(unsigned ArgNo) const {
    assert(hasPaddingArg(ArgNo));
    return ArgInfo[ArgNo].PaddingArgIndex;
  }

  /// Returns index of first IR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getIRArgs(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return std::make_pair(ArgInfo[ArgNo].FirstArgIndex,
                          ArgInfo[ArgNo].NumberOfArgs);
  }

private:
  void construct(const ASTContext &Context, const CodeGen::CGFunctionInfo &FI,
                 bool OnlyRequiredArgs);
};

void ClangToLLVMArgMapping::construct(const ASTContext &Context,
                                      const CodeGen::CGFunctionInfo &FI,
                                      bool OnlyRequiredArgs) {
  unsigned IRArgNo = 0;
  bool SwapThisWithSRet = false;
  const CodeGen::ABIArgInfo &RetAI = FI.getReturnInfo();

#if 0
  // TODO: support function returning struct.
  if (RetAI.getKind() == CodeGen::ABIArgInfo::Indirect) {
    SwapThisWithSRet = RetAI.isSRetAfterThis();
    SRetArgNo = SwapThisWithSRet ? 1 : IRArgNo++;
  }
#endif

  unsigned ArgNo = 0;
  unsigned NumArgs = OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
  for (CodeGen::CGFunctionInfo::const_arg_iterator I = FI.arg_begin();
       ArgNo < NumArgs; ++I, ++ArgNo) {
    assert(I != FI.arg_end());
    QualType ArgType = I->type;
    const CodeGen::ABIArgInfo &AI = I->info;
    // Collect data about IR arguments corresponding to Clang argument ArgNo.
    auto &IRArgs = ArgInfo[ArgNo];

    if (AI.getPaddingType())
      IRArgs.PaddingArgIndex = IRArgNo++;

    switch (AI.getKind()) {
    case CodeGen::ABIArgInfo::Extend:
    case CodeGen::ABIArgInfo::Direct: {
      // FIXME: handle sseregparm someday...
      llvm::StructType *STy = dyn_cast<llvm::StructType>(AI.getCoerceToType());
      if (AI.isDirect() && AI.getCanBeFlattened() && STy) {
        IRArgs.NumberOfArgs = STy->getNumElements();
      } else {
        IRArgs.NumberOfArgs = 1;
      }
      break;
    }
    case CodeGen::ABIArgInfo::Indirect:
    case CodeGen::ABIArgInfo::IndirectAliased:
      IRArgs.NumberOfArgs = 1;
      break;
    case CodeGen::ABIArgInfo::Ignore:
    case CodeGen::ABIArgInfo::InAlloca:
      // ignore and inalloca doesn't have matching LLVM parameters.
      IRArgs.NumberOfArgs = 0;
      break;
    case CodeGen::ABIArgInfo::CoerceAndExpand:
      IRArgs.NumberOfArgs = AI.getCoerceAndExpandTypeSequence().size();
      break;
    case CodeGen::ABIArgInfo::Expand:
      IRArgs.NumberOfArgs = getExpansionSize(ArgType, Context);
      break;
    }

    if (IRArgs.NumberOfArgs > 0) {
      IRArgs.FirstArgIndex = IRArgNo;
      IRArgNo += IRArgs.NumberOfArgs;
    }

    // Skip over the sret parameter when it comes second.  We already handled it
    // above.
    if (IRArgNo == 1 && SwapThisWithSRet)
      IRArgNo++;
  }
  assert(ArgNo == ArgInfo.size());

#if 0
  // TODO: support "usesInAlloca".
  if (FI.usesInAlloca())
    InallocaArgNo = IRArgNo++;
#endif

  TotalIRArgs = IRArgNo;
}
} // namespace

mlir::FunctionType
MLIRASTConsumer::getFunctionType(const CodeGen::CGFunctionInfo &FI) {
  mlir::OpBuilder builder(module->getContext());

  // Compute result type.
  mlir::Type resultType;
  const CodeGen::ABIArgInfo &retAI = FI.getReturnInfo();
  ClangToLLVMArgMapping IRFunctionArgs(CGM.getContext(), FI, true);

  switch (retAI.getKind()) {
  case CodeGen::ABIArgInfo::Expand:
  case CodeGen::ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");

  case CodeGen::ABIArgInfo::Extend:
  case CodeGen::ABIArgInfo::Direct:
    resultType = getMLIRType(FI.getReturnType());
    break;

  case CodeGen::ABIArgInfo::InAlloca:
    if (retAI.getInAllocaSRet()) {
      // sret things on win32 aren't void, they return the sret pointer.
      QualType ret = FI.getReturnType();
      mlir::Type ty = getMLIRType(ret);
      unsigned addressSpace = CGM.getContext().getTargetAddressSpace(ret);
      resultType = mlir::MemRefType::get(-1, ty, {}, addressSpace);
    } else {
      resultType = builder.getNoneType();
    }
    break;

  case CodeGen::ABIArgInfo::Indirect:
    // HACK: remove once we can handle function returning a struct.
    {
      QualType ret = FI.getReturnType();
      mlir::Type ty = getMLIRType(ret);
      unsigned addressSpace = CGM.getContext().getTargetAddressSpace(ret);
      resultType = mlir::MemRefType::get(-1, ty, {}, addressSpace);
      break;
    }
  case CodeGen::ABIArgInfo::Ignore:
    resultType = builder.getNoneType();
    break;

  case CodeGen::ABIArgInfo::CoerceAndExpand:
    assert(false && "TODO");
    break;
  }

  // HACK: We cannot handle (yet) struct returns or inalloca returns so adjust
  // the number of arguments computed in IRFunctionArgs for now.
  unsigned numArgs = IRFunctionArgs.totalIRArgs();
#if 0
  if (IRFunctionArgs.hasSRetArg())
    numArgs--;
  if (IRFunctionArgs.hasInallocaArg())
    numArgs--;
#endif

  SmallVector<mlir::Type, 8> ArgTypes(numArgs);
  LLVM_DEBUG({
    llvm::dbgs() << "IRFunctionArgs.totalIRArgs() = "
                 << IRFunctionArgs.totalIRArgs() << "\n";
    llvm::dbgs() << "numArgs = " << numArgs << "\n";
  });

#if 0
  // TODO
  // Add type for sret argument.
  if (IRFunctionArgs.hasSRetArg()) {
    QualType Ret = FI.getReturnType();
    mlir::Type Ty = getMLIRType(Ret);
    unsigned AddressSpace = CGM.getContext().getTargetAddressSpace(Ret);
    ArgTypes[IRFunctionArgs.getSRetArgNo()] =
        mlir::MemRefType::get(-1, Ty, {}, AddressSpace);
  }

  // Add type for inalloca argument.
  if (IRFunctionArgs.hasInallocaArg()) {
    auto ArgStruct = FI.getArgStruct();
    assert(ArgStruct);

    auto Ty = LLVM::LLVMStructType::getLiteral(module->getContext(), ArgTys);
    ArgTypes[IRFunctionArgs.getInallocaArgNo()] =
        mlir::MemRefType::get(-1, Ty, {}, 0);
  }
#endif

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  CodeGen::CGFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                              ie = it + FI.getNumRequiredArgs();
  for (; it != ie; ++it, ++ArgNo) {
    const QualType &ArgTy = it->type;
    const CodeGen::ABIArgInfo &ArgInfo = it->info;

#if 0
    // TODO
    // Insert a padding type to ensure proper alignment.
    if (IRFunctionArgs.hasPaddingArg(ArgNo))
      ArgTypes[IRFunctionArgs.getPaddingArgNo(ArgNo)] =
          ArgInfo.getPaddingType();
#endif

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

#if 1
    llvm::dbgs() << "ArgNo = " << ArgNo << "\n";
    llvm::dbgs() << "FirstIRArg = " << FirstIRArg << "\n";
    llvm::dbgs() << "NumIRArgs = " << NumIRArgs << "\n";
    llvm::dbgs() << "ArgTy: ";
    ArgTy.dump();
    llvm::dbgs() << "\n";
#endif

    switch (ArgInfo.getKind()) {
    case CodeGen::ABIArgInfo::Ignore:
    case CodeGen::ABIArgInfo::InAlloca:
      assert(NumIRArgs == 0);
      break;
    case CodeGen::ABIArgInfo::Indirect: {
      assert(NumIRArgs == 1);
      llvm::dbgs() << "line " << __LINE__ << "\n";
      // indirect arguments are always on the stack, which is alloca addr space.
      bool ArrayStruct = false;
      mlir::Type argType = getMLIRType(ArgTy, &ArrayStruct);
      if (ArrayStruct)
        argType = getMLIRType(CGM.getContext().getLValueReferenceType(ArgTy));

      ArgTypes[FirstIRArg] = mlir::MemRefType::get(
          -1, argType, {}, CGM.getDataLayout().getAllocaAddrSpace());
      break;
    }
    case CodeGen::ABIArgInfo::IndirectAliased: {
      assert(NumIRArgs == 1);
      llvm::dbgs() << "line " << __LINE__ << "\n";
      bool ArrayStruct = false;
      mlir::Type argType = getMLIRType(ArgTy, &ArrayStruct);
      if (ArrayStruct)
        argType = getMLIRType(CGM.getContext().getLValueReferenceType(ArgTy));

      ArgTypes[FirstIRArg] = mlir::MemRefType::get(
          -1, argType, {}, ArgInfo.getIndirectAddrSpace());
      break;
    }
    case CodeGen::ABIArgInfo::Extend:
    case CodeGen::ABIArgInfo::Direct: {
      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      bool ArrayStruct = false;
      mlir::Type argType = getMLIRType(ArgTy, &ArrayStruct);
      if (ArrayStruct)
        argType = getMLIRType(CGM.getContext().getLValueReferenceType(ArgTy));

      auto st = argType.dyn_cast<mlir::LLVM::LLVMStructType>();
      if (st && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        assert(NumIRArgs == st.getBody().size());
        for (unsigned i = 0, e = st.getBody().size(); i != e; ++i)
          ArgTypes[FirstIRArg + i] = st.getBody()[i];
      } else {
        assert(NumIRArgs == 1);
#if 0
        llvm::dbgs() << "line " << __LINE__ << "\n";
        llvm::dbgs() << "argType: " << argType << "\n";
#endif
        ArgTypes[FirstIRArg] = argType;
      }
      break;
    }
    case CodeGen::ABIArgInfo::CoerceAndExpand:
      assert(false && "TODO");
      break;
    case CodeGen::ABIArgInfo::Expand:
      assert(false && "TODO");
      break;
    }
  }

  SmallVector<mlir::Type, 2> resultTypes;
  if (!resultType.isa<mlir::NoneType>())
    resultTypes.push_back(resultType);

  return builder.getFunctionType(ArgTypes, resultTypes);
}

mlir::FunctionOpInterface
MLIRASTConsumer::createMLIRFunction(const FunctionToEmit &FTE,
                                    std::string mangledName, bool IsThunk,
                                    bool ShouldEmit) {
  const FunctionDecl &FD = FTE.getDecl();
  Location loc = getMLIRLocation(FD.getLocation());
  mlir::OpBuilder builder(module->getContext());

#if 1
  GlobalDecl GD;
  if (auto CC = dyn_cast<CXXConstructorDecl>(&FD))
    GD = GlobalDecl(CC, CXXCtorType::Ctor_Complete);
  else if (auto CC = dyn_cast<CXXDestructorDecl>(&FD))
    GD = GlobalDecl(CC, CXXDtorType::Dtor_Complete);
  else
    GD = GlobalDecl(&FD);

  const CodeGen::CGFunctionInfo &FI =
      CGM.getTypes().arrangeGlobalDeclaration(GD);
  mlir::FunctionType funcTy = getFunctionType(FI);
#endif
#if 1
  SmallVector<CodeGenUtils::ParmDesc, 4> parmDescriptors;
  createMLIRParameterDescriptors(FD, parmDescriptors);

  SmallVector<CodeGenUtils::ResultDesc, 4> resDescriptors;
  createMLIRResultDescriptors(FD, resDescriptors);

  SmallVector<mlir::Type, 4> parmTypes, retTypes;
  CodeGenUtils::ParmDesc::getTypes(parmDescriptors, parmTypes);
  CodeGenUtils::ResultDesc::getTypes(resDescriptors, retTypes);

  auto funcTy1 = builder.getFunctionType(parmTypes, retTypes);
  if (funcTy != funcTy1) {
    llvm::dbgs() << "BINGO - types are different for " << mangledName
                 << "!!!\n";
    llvm::dbgs() << "New funcTy: " << funcTy << "\n";
    llvm::dbgs() << "Old funcTy1: " << funcTy1 << "\n";
  }
  if (UseOldFunctionType)
    funcTy = funcTy1;
#endif

  mlir::FunctionOpInterface function =
      FD.hasAttr<SYCLKernelAttr>()
          ? builder.create<gpu::GPUFuncOp>(loc, mangledName, funcTy)
          : builder.create<func::FuncOp>(loc, mangledName, funcTy);

  setMLIRFunctionVisibility(function, FTE, ShouldEmit);
  setMLIRFunctionAttributes(function, FTE, FI, IsThunk, ShouldEmit);
#if 0
  setMLIRFunctionParmsAttributes(function, parmDescriptors);
  setMLIRFunctionResultAttributes(function, resDescriptors);
#endif

  /// Inject the MLIR function created in either the device module or in the
  /// host module, depending on the calling context.
  switch (FTE.getContext()) {
  case FunctionContext::Host:
    module->push_back(function);
    functions[mangledName] = cast<func::FuncOp>(function);
    break;
  case FunctionContext::SYCLDevice:
    getDeviceModule(*module).push_back(function);
    deviceFunctions[mangledName] = function;
    break;
  }

  LLVM_DEBUG(llvm::dbgs() << "Created MLIR function: " << function << "\n");

  return function;
}

void MLIRASTConsumer::createMLIRResultDescriptors(
    const clang::FunctionDecl &FD,
    llvm::SmallVectorImpl<CodeGenUtils::ResultDesc> &resDescriptors) {
  assert(resDescriptors.empty() && "Expecting 'resDescriptors' to be empty");

  bool isArrayReturn = false;
  getMLIRType(FD.getReturnType(), &isArrayReturn);

  if (!isArrayReturn) {
    auto rt = getMLIRType(FD.getReturnType());
    if (!rt.isa<mlir::NoneType>())
      resDescriptors.push_back(rt);
  }
}

void MLIRASTConsumer::setMLIRFunctionVisibility(
    mlir::FunctionOpInterface function, const FunctionToEmit &FTE,
    bool shouldEmit) {
  const FunctionDecl &FD = FTE.getDecl();
  SymbolTable::Visibility visibility = SymbolTable::Visibility::Public;

  if (!shouldEmit || !FD.isDefined() || FD.hasAttr<CUDAGlobalAttr>() ||
      FD.hasAttr<CUDADeviceAttr>())
    visibility = SymbolTable::Visibility::Private;
  else {
    llvm::GlobalValue::LinkageTypes LV = getLLVMLinkageType(FD, shouldEmit);
    if (LV == llvm::GlobalValue::InternalLinkage ||
        LV == llvm::GlobalValue::PrivateLinkage)
      visibility = SymbolTable::Visibility::Private;
  }

  SymbolTable::setSymbolVisibility(function, visibility);
}

void MLIRASTConsumer::constructAttributeList(const CodeGen::CGFunctionInfo &FI,
                                             CodeGen::CGCalleeInfo CalleeInfo,
                                             mlirclang::AttributeList &AttrList,
                                             bool AttrOnCallSite,
                                             bool IsThunk) {
  MLIRContext *ctx = module->getContext();
  mlirclang::AttrBuilder funcAttrsBuilder(*ctx);
  mlirclang::AttrBuilder retAttrsBuilder(*ctx);

  unsigned CC = FI.getEffectiveCallingConvention();
  funcAttrsBuilder.addAttribute(
      "llvm.cconv", mlir::LLVM::CConvAttr::get(
                        ctx, static_cast<mlir::LLVM::cconv::CConv>(CC)));

  if (FI.isNoReturn())
    funcAttrsBuilder.addAttribute(llvm::Attribute::NoReturn);
  if (FI.isCmseNSCall())
    funcAttrsBuilder.addAttribute("cmse_nonsecure_call", UnitAttr::get(ctx));

  // Collect function IR attributes from the callee prototype if we have one.
  addAttributesFromFunctionProtoType(funcAttrsBuilder,
                                     CalleeInfo.getCalleeFunctionProtoType());

  const Decl *targetDecl = CalleeInfo.getCalleeDecl().getDecl();

  // Attach assumption attributes to the declaration. If this is a call
  // site, attach assumptions from the caller to the call as well.
  addAttributesFromAssumes(funcAttrsBuilder, targetDecl);

  bool HasOptnone = false;
  // The NoBuiltinAttr attached to the target FunctionDecl.
  const NoBuiltinAttr *NBA = nullptr;

  // Collect function IR attributes based on declaration-specific
  // information.
  if (targetDecl) {
    if (targetDecl->hasAttr<ReturnsTwiceAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::ReturnsTwice);
    if (targetDecl->hasAttr<NoThrowAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoUnwind);
    if (targetDecl->hasAttr<NoReturnAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoReturn);
    if (targetDecl->hasAttr<ColdAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::Cold);
    if (targetDecl->hasAttr<HotAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::Hot);
    if (targetDecl->hasAttr<NoDuplicateAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoDuplicate);
    if (targetDecl->hasAttr<ConvergentAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::Convergent);

    if (const FunctionDecl *Fn = dyn_cast<FunctionDecl>(targetDecl)) {
      addAttributesFromFunctionProtoType(
          funcAttrsBuilder, Fn->getType()->getAs<FunctionProtoType>());
      if (AttrOnCallSite && Fn->isReplaceableGlobalAllocationFunction()) {
        // A sane operator new returns a non-aliasing pointer.
        auto Kind = Fn->getDeclName().getCXXOverloadedOperator();
        if (CGM.getCodeGenOpts().AssumeSaneOperatorNew &&
            (Kind == OO_New || Kind == OO_Array_New))
          retAttrsBuilder.addAttribute(llvm::Attribute::NoAlias);
      }
      const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Fn);
      const bool isVirtualCall = MD && MD->isVirtual();
      // Don't use [[noreturn]], _Noreturn or [[no_builtin]] for a call to a
      // virtual function. These attributes are not inherited by overloads.
      if (!(AttrOnCallSite && isVirtualCall)) {
        if (Fn->isNoReturn())
          funcAttrsBuilder.addAttribute(llvm::Attribute::NoReturn);
        NBA = Fn->getAttr<NoBuiltinAttr>();
      }
      // Only place nomerge attribute on call sites, never functions. This
      // allows it to work on indirect virtual function calls.
      if (AttrOnCallSite && targetDecl->hasAttr<NoMergeAttr>())
        funcAttrsBuilder.addAttribute(llvm::Attribute::NoMerge);
    }

    // 'const', 'pure' and 'noalias' attributed functions are also nounwind.
    if (targetDecl->hasAttr<ConstAttr>()) {
      funcAttrsBuilder.addAttribute(llvm::Attribute::ReadNone);
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'const' functions have greater restrictions than
      // 'pure' functions, so they also cannot have infinite loops.
      funcAttrsBuilder.addAttribute(llvm::Attribute::WillReturn);
    } else if (targetDecl->hasAttr<PureAttr>()) {
      funcAttrsBuilder.addAttribute(llvm::Attribute::ReadOnly);
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'pure' functions cannot have infinite loops.
      funcAttrsBuilder.addAttribute(llvm::Attribute::WillReturn);
    } else if (targetDecl->hasAttr<NoAliasAttr>()) {
      funcAttrsBuilder.addAttribute(llvm::Attribute::ArgMemOnly);
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoUnwind);
    }
    if (targetDecl->hasAttr<RestrictAttr>())
      retAttrsBuilder.addAttribute(llvm::Attribute::NoAlias);
    if (targetDecl->hasAttr<ReturnsNonNullAttr>() &&
        !CGM.getCodeGenOpts().NullPointerIsValid)
      retAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
    if (targetDecl->hasAttr<AnyX86NoCallerSavedRegistersAttr>())
      funcAttrsBuilder.addAttribute("no_caller_saved_registers",
                                    UnitAttr::get(ctx));
    if (targetDecl->hasAttr<AnyX86NoCfCheckAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoCfCheck);
    if (targetDecl->hasAttr<LeafAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::NoCallback);

    HasOptnone = targetDecl->hasAttr<OptimizeNoneAttr>();
    if (auto *AllocSize = targetDecl->getAttr<AllocSizeAttr>()) {
      Optional<unsigned> NumElemsParam;
      if (AllocSize->getNumElemsParam().isValid())
        NumElemsParam = AllocSize->getNumElemsParam().getLLVMIndex();
      uint64_t RawArgs = packAllocSizeArgs(
          AllocSize->getElemSizeParam().getLLVMIndex(), NumElemsParam);
      funcAttrsBuilder.addAttribute(llvm::Attribute::AllocSize, RawArgs);
    }

    if (targetDecl->hasAttr<OpenCLKernelAttr>()) {
      if (CGM.getLangOpts().OpenCLVersion <= 120) {
        // OpenCL v1.2 Work groups are always uniform
        funcAttrsBuilder.addAttribute("uniform-work-group-size",
                                      StringAttr::get(ctx, "true"));
      } else {
        // OpenCL v2.0 Work groups may be whether uniform or not.
        // '-cl-uniform-work-group-size' compile option gets a hint
        // to the compiler that the global work-size be a multiple of
        // the work-group size specified to clEnqueueNDRangeKernel
        // (i.e. work groups are uniform).
        funcAttrsBuilder.addAttribute(
            "uniform-work-group-size",
            StringAttr::get(
                ctx, llvm::toStringRef(CGM.getCodeGenOpts().UniformWGSize)));
      }
    }
  }

  // Attach "no-builtins" attributes to:
  // * call sites: both `nobuiltin` and "no-builtins" or "no-builtin-<name>".
  // * definitions: "no-builtins" or "no-builtin-<name>" only.
  // The attributes can come from:
  // * LangOpts: -ffreestanding, -fno-builtin, -fno-builtin-<name>
  // * FunctionDecl attributes: __attribute__((no_builtin(...)))
  addNoBuiltinAttributes(funcAttrsBuilder, CGM.getLangOpts(), NBA);

// Collect function IR attributes based on global settiings.
// TODO
#if 0
  getDefaultFunctionAttributes(Name, HasOptnone, AttrOnCallSite, funcAttrsBuilder);
#endif

  // Override some default IR attributes based on declaration-specific
  // information.
  if (targetDecl) {
    if (targetDecl->hasAttr<NoSpeculativeLoadHardeningAttr>())
      funcAttrsBuilder.removeAttribute(
          llvm::Attribute::SpeculativeLoadHardening);
    if (targetDecl->hasAttr<SpeculativeLoadHardeningAttr>())
      funcAttrsBuilder.addAttribute(llvm::Attribute::SpeculativeLoadHardening);
    if (targetDecl->hasAttr<NoSplitStackAttr>())
      funcAttrsBuilder.removeAttribute("split-stack");
    if (targetDecl->hasAttr<ZeroCallUsedRegsAttr>()) {
      // A function "__attribute__((...))" overrides the command-line flag.
      auto Kind =
          targetDecl->getAttr<ZeroCallUsedRegsAttr>()->getZeroCallUsedRegs();
      funcAttrsBuilder.removeAttribute("zero-call-used-regs");
      funcAttrsBuilder.addAttribute(
          "zero-call-used-regs",
          StringAttr::get(
              ctx,
              ZeroCallUsedRegsAttr::ConvertZeroCallUsedRegsKindToStr(Kind)));
    }

    // Add NonLazyBind attribute to function declarations when -fno-plt is used.
    if (CGM.getCodeGenOpts().NoPLT) {
      if (auto *Fn = dyn_cast<FunctionDecl>(targetDecl)) {
        if (!Fn->isDefined() && !AttrOnCallSite) {
          funcAttrsBuilder.addAttribute(llvm::Attribute::NonLazyBind);
        }
      }
    }
  }

  // Add "sample-profile-suffix-elision-policy" attribute for internal linkage
  // functions with -funique-internal-linkage-names.
  if (targetDecl && CGM.getCodeGenOpts().UniqueInternalLinkageNames) {
    if (isa<FunctionDecl>(targetDecl)) {
      if (CGM.getFunctionLinkage(CalleeInfo.getCalleeDecl()) ==
          llvm::GlobalValue::InternalLinkage)
        funcAttrsBuilder.addAttribute("sample-profile-suffix-elision-policy",
                                      StringAttr::get(ctx, "selected"));
    }
  }

  // Collect non-call-site function IR attributes from declaration-specific
  // information.
  if (!AttrOnCallSite) {
    if (targetDecl && targetDecl->hasAttr<CmseNSEntryAttr>())
      funcAttrsBuilder.addAttribute("cmse_nonsecure_entry", UnitAttr::get(ctx));

    // Whether tail calls are enabled.
    auto shouldDisableTailCalls = [&] {
      // Should this be honored in getDefaultFunctionAttributes?
      if (CGM.getCodeGenOpts().DisableTailCalls)
        return true;

      if (!targetDecl)
        return false;

      if (targetDecl->hasAttr<DisableTailCallsAttr>() ||
          targetDecl->hasAttr<AnyX86InterruptAttr>())
        return true;

      if (CGM.getCodeGenOpts().NoEscapingBlockTailCalls) {
        if (const auto *BD = dyn_cast<BlockDecl>(targetDecl))
          if (!BD->doesNotEscape())
            return true;
      }

      return false;
    };
    if (shouldDisableTailCalls())
      funcAttrsBuilder.addAttribute("disable-tail-calls",
                                    StringAttr::get(ctx, "true"));

      // TODO
#if 0
    // CPU/feature overrides.  addDefaultFunctionDefinitionAttributes
    // handles these separately to set them based on the global defaults.
    getCPUAndFeaturesAttributes(CalleeInfo.getCalleeDecl(), funcAttrsBuilder);
#endif
  }

  // Collect attributes from arguments and return values.
  ClangToLLVMArgMapping IRFunctionArgs(CGM.getContext(), FI);

  QualType RetTy = FI.getReturnType();
  const CodeGen::ABIArgInfo &RetAI = FI.getReturnInfo();
  const llvm::DataLayout &DL = CGM.getDataLayout();

  // C++ explicitly makes returning undefined values UB. C's rule only applies
  // to used values, so we never mark them noundef for now.
  bool HasStrictReturn = CGM.getLangOpts().CPlusPlus;
  if (targetDecl && HasStrictReturn) {
    if (const FunctionDecl *FDecl = dyn_cast<FunctionDecl>(targetDecl))
      HasStrictReturn &= !FDecl->isExternC();
    else if (const VarDecl *VDecl = dyn_cast<VarDecl>(targetDecl))
      // Function pointer
      HasStrictReturn &= !VDecl->isExternC();
  }

  // We don't want to be too aggressive with the return checking, unless
  // it's explicit in the code opts or we're using an appropriate sanitizer.
  // Try to respect what the programmer intended.
  HasStrictReturn &= CGM.getCodeGenOpts().StrictReturn ||
                     !CGM.MayDropFunctionReturn(CGM.getContext(), RetTy) ||
                     CGM.getLangOpts().Sanitize.has(SanitizerKind::Memory) ||
                     CGM.getLangOpts().Sanitize.has(SanitizerKind::Return);

  // Determine if the return type could be partially undef
  if (CGM.getCodeGenOpts().EnableNoundefAttrs && HasStrictReturn) {
    if (!RetTy->isVoidType() &&
        RetAI.getKind() != CodeGen::ABIArgInfo::Indirect &&
        CodeGenUtils::determineNoUndef(RetTy, CGM.getTypes(), DL, RetAI))
      retAttrsBuilder.addAttribute(llvm::Attribute::NoUndef);
  }

  switch (RetAI.getKind()) {
  case CodeGen::ABIArgInfo::Extend:
    if (RetAI.isSignExt())
      retAttrsBuilder.addAttribute(llvm::Attribute::SExt);
    else
      retAttrsBuilder.addAttribute(llvm::Attribute::ZExt);
    LLVM_FALLTHROUGH;
  case CodeGen::ABIArgInfo::Direct:
    if (RetAI.getInReg())
      retAttrsBuilder.addAttribute(llvm::Attribute::InReg);
    break;
  case CodeGen::ABIArgInfo::Ignore:
    break;

  case CodeGen::ABIArgInfo::InAlloca:
  case CodeGen::ABIArgInfo::Indirect: {
    // inalloca and sret disable readnone and readonly
    funcAttrsBuilder.removeAttribute(llvm::Attribute::ReadOnly)
        .removeAttribute(llvm::Attribute::ReadNone);
    break;
  }

  case CodeGen::ABIArgInfo::CoerceAndExpand:
    break;

  case CodeGen::ABIArgInfo::Expand:
  case CodeGen::ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");
  }

  if (!IsThunk) {
    // FIXME: fix this properly, https://reviews.llvm.org/D100388
    if (const auto *refTy = RetTy->getAs<ReferenceType>()) {
      QualType PTy = refTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        retAttrsBuilder.addAttribute(
            llvm::Attribute::Dereferenceable,
            CGM.getMinimumObjectSize(PTy).getQuantity());
      if (CGM.getContext().getTargetAddressSpace(PTy) == 0 &&
          !CGM.getCodeGenOpts().NullPointerIsValid)
        retAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
      if (PTy->isObjectType()) {
        llvm::Align Alignment =
            CGM.getNaturalPointeeTypeAlignment(RetTy).getAsAlign();
        retAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                     Alignment.value());
      }
    }
  }

  bool hasUsedSRet = false;
  SmallVector<mlir::NamedAttrList, 4> ArgAttrs(IRFunctionArgs.totalIRArgs());

  // Attach attributes to sret.
  if (IRFunctionArgs.hasSRetArg()) {
    mlirclang::AttrBuilder SRETAttrs(*ctx);
#if 0
    // TODO
    auto val = mlir::Attribute::get(ctx, llvm::Attribute::StructRet,
                                    getMLIRType(RetTy));
    SRETAttrs.addAttribute(llvm::Attribute::StructRet, val);
#endif
    hasUsedSRet = true;
    if (RetAI.getInReg())
      SRETAttrs.addAttribute(llvm::Attribute::InReg);
    SRETAttrs.addAttribute(llvm::Attribute::Alignment,
                           RetAI.getIndirectAlign().getQuantity());
    ArgAttrs[IRFunctionArgs.getSRetArgNo()] = SRETAttrs.getAttrs();
  }

  // Attach attributes to inalloca argument.
  if (IRFunctionArgs.hasInallocaArg()) {
    mlirclang::AttrBuilder Attrs(*ctx);
#if 0
    // TODO
    Attrs.addInAllocaAttr(FI.getArgStruct());
#endif
    ArgAttrs[IRFunctionArgs.getInallocaArgNo()] = Attrs.getAttrs();
  }

  // Apply `nonnull`, `dereferencable(N)` and `align N` to the `this` argument,
  // unless this is a thunk function.
  // FIXME: fix this properly, https://reviews.llvm.org/D100388
  if (FI.isInstanceMethod() && !IRFunctionArgs.hasInallocaArg() &&
      !FI.arg_begin()->type->isVoidPointerType() && !IsThunk) {
    auto IRArgs = IRFunctionArgs.getIRArgs(0);

    assert(IRArgs.second == 1 && "Expected only a single `this` pointer.");

    mlirclang::AttrBuilder attrsBuilder(*ctx);

    QualType ThisTy =
        FI.arg_begin()->type.castAs<clang::PointerType>()->getPointeeType();

    if (!CGM.getCodeGenOpts().NullPointerIsValid &&
        CGM.getContext().getTargetAddressSpace(FI.arg_begin()->type) == 0) {
      attrsBuilder.addAttribute(llvm::Attribute::NonNull);
      attrsBuilder.addAttribute(llvm::Attribute::Dereferenceable,
                                CGM.getMinimumObjectSize(ThisTy).getQuantity());
    } else {
      // FIXME dereferenceable should be correct here, regardless of
      // NullPointerIsValid. However, dereferenceable currently does not always
      // respect NullPointerIsValid and may imply nonnull and break the program.
      // See https://reviews.llvm.org/D66618 for discussions.
      attrsBuilder.addAttribute(
          llvm::Attribute::DereferenceableOrNull,
          CGM.getMinimumObjectSize(FI.arg_begin()
                                       ->type.castAs<clang::PointerType>()
                                       ->getPointeeType())
              .getQuantity());
    }

    llvm::Align Alignment =
        CGM.getNaturalTypeAlignment(ThisTy, /*BaseInfo=*/nullptr,
                                    /*TBAAInfo=*/nullptr,
                                    /*forPointeeType=*/true)
            .getAsAlign();
    attrsBuilder.addAttribute(llvm::Attribute::Alignment, Alignment.value());
    ArgAttrs[IRArgs.first] = attrsBuilder.getAttrs();
  }

  unsigned ArgNo = 0;
  for (CodeGen::CGFunctionInfo::const_arg_iterator I = FI.arg_begin(),
                                                   E = FI.arg_end();
       I != E; ++I, ++ArgNo) {
    QualType ParamType = I->type;
    const CodeGen::ABIArgInfo &AI = I->info;
    mlirclang::AttrBuilder attrsBuilder(*ctx);

    // Add attribute for padding argument, if necessary.
    if (IRFunctionArgs.hasPaddingArg(ArgNo)) {
      if (AI.getPaddingInReg()) {
        ArgAttrs[IRFunctionArgs.getPaddingArgNo(ArgNo)] =
            mlirclang::AttrBuilder(*ctx)
                .addAttribute(llvm::Attribute::InReg)
                .getAttrs();
      }
    }

    // Decide whether the argument we're handling could be partially undef
    if (CGM.getCodeGenOpts().EnableNoundefAttrs &&
        CodeGenUtils::determineNoUndef(ParamType, CGM.getTypes(), DL, AI)) {
      attrsBuilder.addAttribute(llvm::Attribute::NoUndef);
    }

    // 'restrict' -> 'noalias' is done in EmitFunctionProlog when we
    // have the corresponding parameter variable.  It doesn't make
    // sense to do it here because parameters are so messed up.
    switch (AI.getKind()) {
    case CodeGen::ABIArgInfo::Extend:
      if (AI.isSignExt())
        attrsBuilder.addAttribute(llvm::Attribute::SExt);
      else
        attrsBuilder.addAttribute(llvm::Attribute::ZExt);
      LLVM_FALLTHROUGH;
    case CodeGen::ABIArgInfo::Direct:
      if (ArgNo == 0 && FI.isChainCall())
        attrsBuilder.addAttribute(llvm::Attribute::Nest);
      else if (AI.getInReg())
        attrsBuilder.addAttribute(llvm::Attribute::InReg);
      attrsBuilder.addAttribute(llvm::Attribute::StackAlignment,
                                AI.getDirectAlign());
      break;

    case CodeGen::ABIArgInfo::Indirect: {
      if (AI.getInReg())
        attrsBuilder.addAttribute(llvm::Attribute::InReg);

#if 0
// TODO
      if (AI.getIndirectByVal())
        attrsBuilder.addByValAttr(CGM.getTypes().ConvertTypeForMem(ParamType));
#endif

      auto *Decl = ParamType->getAsRecordDecl();
      if (CGM.getCodeGenOpts().PassByValueIsNoAlias && Decl &&
          Decl->getArgPassingRestrictions() == RecordDecl::APK_CanPassInRegs)
        // When calling the function, the pointer passed in will be the only
        // reference to the underlying object. Mark it accordingly.
        attrsBuilder.addAttribute(llvm::Attribute::NoAlias);

      // TODO: We could add the byref attribute if not byval, but it would
      // require updating many testcases.

      CharUnits Align = AI.getIndirectAlign();

      // In a byval argument, it is important that the required
      // alignment of the type is honored, as LLVM might be creating a
      // *new* stack object, and needs to know what alignment to give
      // it. (Sometimes it can deduce a sensible alignment on its own,
      // but not if clang decides it must emit a packed struct, or the
      // user specifies increased alignment requirements.)
      //
      // This is different from indirect *not* byval, where the object
      // exists already, and the align attribute is purely
      // informative.
      assert(!Align.isZero());

      // For now, only add this when we have a byval argument.
      // TODO: be less lazy about updating test cases.
      if (AI.getIndirectByVal())
        attrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                  Align.getQuantity());

      // byval disables readnone and readonly.
      funcAttrsBuilder.removeAttribute(llvm::Attribute::ReadOnly)
          .removeAttribute(llvm::Attribute::ReadNone);

      break;
    }
    case CodeGen::ABIArgInfo::IndirectAliased: {
      CharUnits Align = AI.getIndirectAlign();
#if 0
      //TODO
      attrsBuilder.addByRefAttr(CGM.getTypes().ConvertTypeForMem(ParamType));
#endif
      attrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                Align.getQuantity());
      break;
    }
    case CodeGen::ABIArgInfo::Ignore:
    case CodeGen::ABIArgInfo::Expand:
    case CodeGen::ABIArgInfo::CoerceAndExpand:
      break;

    case CodeGen::ABIArgInfo::InAlloca:
      // inalloca disables readnone and readonly.
      funcAttrsBuilder.removeAttribute(llvm::Attribute::ReadOnly)
          .removeAttribute(llvm::Attribute::ReadNone);
      continue;
    }

    if (const auto *RefTy = ParamType->getAs<ReferenceType>()) {
      QualType PTy = RefTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        attrsBuilder.addAttribute(llvm::Attribute::Dereferenceable,
                                  CGM.getMinimumObjectSize(PTy).getQuantity());
      if (CGM.getContext().getTargetAddressSpace(PTy) == 0 &&
          !CGM.getCodeGenOpts().NullPointerIsValid)
        attrsBuilder.addAttribute(llvm::Attribute::NonNull);
      if (PTy->isObjectType()) {
        llvm::Align Alignment =
            CGM.getNaturalPointeeTypeAlignment(ParamType).getAsAlign();
        attrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                  Alignment.value());
      }
    }

    // From OpenCL spec v3.0.10 section 6.3.5 Alignment of Types:
    // > For arguments to a __kernel function declared to be a pointer to a
    // > data type, the OpenCL compiler can assume that the pointee is always
    // > appropriately aligned as required by the data type.
    if (targetDecl && targetDecl->hasAttr<OpenCLKernelAttr>() &&
        ParamType->isPointerType()) {
      QualType PTy = ParamType->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType()) {
        CharUnits Alignment = CGM.getNaturalPointeeTypeAlignment(ParamType);
        attrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                  Alignment.getQuantity());
      }
    }

    switch (FI.getExtParameterInfo(ArgNo).getABI()) {
    case ParameterABI::Ordinary:
      break;

    case ParameterABI::SwiftIndirectResult: {
      // Add 'sret' if we haven't already used it for something, but
      // only if the result is void.
#if 0      
      if (!hasUsedSRet && RetTy->isVoidType()) {
        //TODO
        attrsBuilder.addStructRetAttr(getTypes().ConvertTypeForMem(ParamType));
        hasUsedSRet = true;
    }
#endif
      // Add 'noalias' in either case.
      attrsBuilder.addAttribute(llvm::Attribute::NoAlias);

      // Add 'dereferenceable' and 'alignment'.
      auto PTy = ParamType->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType()) {
        auto info = CGM.getContext().getTypeInfoInChars(PTy);
        attrsBuilder.addAttribute(llvm::Attribute::Dereferenceable,
                                  info.Width.getQuantity());
        attrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                  info.Align.getQuantity());
      }
      break;
    }

    case ParameterABI::SwiftErrorResult:
      attrsBuilder.addAttribute(llvm::Attribute::SwiftError);
      break;

    case ParameterABI::SwiftContext:
      attrsBuilder.addAttribute(llvm::Attribute::SwiftSelf);
      break;

    case ParameterABI::SwiftAsyncContext:
      attrsBuilder.addAttribute(llvm::Attribute::SwiftAsync);
      break;
    }

    if (FI.getExtParameterInfo(ArgNo).isNoEscape())
      attrsBuilder.addAttribute(llvm::Attribute::NoCapture);

    if (attrsBuilder.hasAttributes()) {
      unsigned FirstIRArg, NumIRArgs;
      std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);
      for (unsigned i = 0; i < NumIRArgs; i++)
        ArgAttrs[FirstIRArg + i].append(attrsBuilder.getAttrs());
    }
  }
  assert(ArgNo == FI.arg_size());

  AttrList.addAttrs(funcAttrsBuilder, retAttrsBuilder, ArgAttrs);
}

void MLIRASTConsumer::setMLIRFunctionAttributes(
    mlir::FunctionOpInterface function, const FunctionToEmit &FTE,
    const CodeGen::CGFunctionInfo &FI, bool IsThunk, bool ShouldEmit) {
  using Attribute = llvm::Attribute;

  const FunctionDecl &FD = FTE.getDecl();
  MLIRContext *ctx = module->getContext();
  mlirclang::AttrBuilder attrBuilder(*ctx);

  LLVM::Linkage lnk = getMLIRLinkage(getLLVMLinkageType(FD, ShouldEmit));
  attrBuilder.addAttribute("llvm.linkage",
                           mlir::LLVM::LinkageAttr::get(ctx, lnk));

  bool isDeviceContext = (FTE.getContext() == FunctionContext::SYCLDevice);
  if (!isDeviceContext) {
    LLVM_DEBUG(llvm::dbgs()
               << "Not in a device context - skipping setting attributes for "
               << FD.getNameAsString() << "\n");

    // HACK: we want to avoid setting additional attributes on non-sycl
    // functions because we do not want to adjust the test cases at this time
    // (if we did we would have merge conflicts if we ever update polygeist).
    NamedAttrList attrs(function->getAttrDictionary());
    attrs.append(attrBuilder.getAttrs());
    function->setAttrs(attrs.getDictionary(module->getContext()));
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Setting attributes for " << FD.getNameAsString()
                          << "\n");

#if 0
  const auto *FPT = FD.getType()->getAs<FunctionProtoType>();
  CodeGen::CGCalleeInfo CalleeInfo(FPT);

  mlirclang::AttributeList PAL;
  constructAttributeList(FI, CalleeInfo, PAL, /*AttrOnCallSite=*/false,
                         IsThunk);

  NamedAttrList attrs(function->getAttrDictionary());
  attrs.append(PAL.getFnAttrs());
  function->setAttrs(attrs.getDictionary(ctx));
#else

  // Mark a SYCL kernel as a SPIRV kernel.
  if (FD.hasAttr<SYCLKernelAttr>()) {
    attrBuilder
        .addAttribute(gpu::GPUDialect::getKernelFuncAttrName(),
                      UnitAttr::get(ctx))
        .addPassThroughAttribute(ArrayAttr::get(
            ctx, {StringAttr::get(ctx, "sycl-module-id"),
                  StringAttr::get(ctx, llvmMod.getModuleIdentifier())}));
  }

  // Calling conventions for SPIRV functions.
  attrBuilder.addAttribute("llvm.cconv",
                           mlir::LLVM::CConvAttr::get(
                               ctx, FD.hasAttr<SYCLKernelAttr>()
                                        ? mlir::LLVM::cconv::CConv::SPIR_KERNEL
                                        : mlir::LLVM::cconv::CConv::SPIR_FUNC));

  // SYCL v1.2.1 s3.10:
  //   kernels and device function cannot include RTTI information,
  //   exception classes, recursive code, virtual functions or make use of
  //   C++ libraries that are not compiled for the device.
  attrBuilder.addPassThroughAttribute(Attribute::AttrKind::NoRecurse)
      .addPassThroughAttribute(Attribute::AttrKind::NoUnwind);

  if (CGM.getLangOpts().assumeFunctionsAreConvergent())
    attrBuilder.addPassThroughAttribute(Attribute::AttrKind::Convergent);

  auto functionMustProgress = [](const CodeGen::CodeGenModule &CGM) -> bool {
    if (CGM.getCodeGenOpts().getFiniteLoops() ==
        CodeGenOptions::FiniteLoopsKind::Never)
      return false;
    return CGM.getLangOpts().CPlusPlus11;
  };

  if (functionMustProgress(CGM))
    attrBuilder.addPassThroughAttribute(Attribute::AttrKind::MustProgress);

  NamedAttrList attrs(function->getAttrDictionary());
  attrs.append(attrBuilder.getAttrs());
  function->setAttrs(attrs.getDictionary(ctx));
#endif
}

void MLIRASTConsumer::setMLIRFunctionParmsAttributes(
    mlir::FunctionOpInterface function,
    const SmallVectorImpl<CodeGenUtils::ParmDesc> &parmDescriptors) const {
  assert(function.getNumArguments() == parmDescriptors.size() && "Mismatch");

  SmallVector<mlir::NamedAttrList, 4> parmAttrs;
  CodeGenUtils::ParmDesc::getAttributes(parmDescriptors, parmAttrs);

  for (unsigned argIndex : llvm::seq<unsigned>(0, function.getNumArguments()))
    for (mlir::NamedAttribute attr : parmAttrs[argIndex])
      function.setArgAttr(argIndex, attr.getName(), attr.getValue());
}

void MLIRASTConsumer::setMLIRFunctionResultAttributes(
    mlir::FunctionOpInterface function,
    const SmallVectorImpl<CodeGenUtils::ResultDesc> &resDescriptors) const {
  assert(function.getNumResults() == resDescriptors.size() && "Mismatch");

  SmallVector<mlir::NamedAttrList, 2> resAttrs;
  CodeGenUtils::ResultDesc::getAttributes(resDescriptors, resAttrs);

  for (unsigned resIndex : llvm::seq<unsigned>(0, function.getNumResults()))
    for (mlir::NamedAttribute attr : resAttrs[resIndex])
      function.setResultAttr(resIndex, attr.getName(), attr.getValue());
}

llvm::Optional<mlir::FunctionOpInterface>
MLIRASTConsumer::getMLIRFunction(const std::string &mangledName,
                                 FunctionContext context) const {
  const auto find = [&](const auto &map) {
    const auto Iter = map.find(mangledName);
    return Iter == map.end()
               ? llvm::None
               : llvm::Optional<mlir::FunctionOpInterface>{Iter->second};
  };
  switch (context) {
  case FunctionContext::Host:
    return find(functions);
  case FunctionContext::SYCLDevice:
    return find(deviceFunctions);
  }
  llvm_unreachable("Invalid function context");
}

/// Iteratively get the size of each dim of the given ConstantArrayType inst.
static void getConstantArrayShapeAndElemType(const clang::QualType &ty,
                                             SmallVectorImpl<int64_t> &shape,
                                             clang::QualType &elemTy) {
  shape.clear();

  clang::QualType curTy = ty;
  while (curTy->isConstantArrayType()) {
    auto cstArrTy = cast<clang::ConstantArrayType>(curTy);
    shape.push_back(cstArrTy->getSize().getSExtValue());
    curTy = cstArrTy->getElementType();
  }

  elemTy = curTy;
}

mlir::Type MLIRASTConsumer::getMLIRType(clang::QualType qt, bool *implicitRef,
                                        bool allowMerge) {
  if (auto ET = dyn_cast<clang::ElaboratedType>(qt)) {
    return getMLIRType(ET->getNamedType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::UsingType>(qt)) {
    return getMLIRType(ET->getUnderlyingType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::ParenType>(qt)) {
    return getMLIRType(ET->getInnerType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::DeducedType>(qt)) {
    return getMLIRType(ET->getDeducedType(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::SubstTemplateTypeParmType>(qt)) {
    return getMLIRType(ST->getReplacementType(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::TemplateSpecializationType>(qt)) {
    return getMLIRType(ST->desugar(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::TypedefType>(qt)) {
    return getMLIRType(ST->desugar(), implicitRef, allowMerge);
  }
  if (auto DT = dyn_cast<clang::DecltypeType>(qt)) {
    return getMLIRType(DT->desugar(), implicitRef, allowMerge);
  }

  if (auto DT = dyn_cast<clang::DecayedType>(qt)) {
    bool assumeRef = false;
    auto mlirty = getMLIRType(DT->getOriginalType(), &assumeRef, allowMerge);
    if (memRefABI && assumeRef) {
      // Constant array types like `int A[30][20]` will be converted to LLVM
      // type `[20 x i32]* %0`, which has the outermost dimension size erased,
      // and we can only recover to `memref<?x20xi32>` from there. This
      // prevents us from doing more comprehensive analysis. Here we
      // specifically handle this case by unwrapping the clang-adjusted type,
      // to get the corresponding ConstantArrayType with the full dimensions.
      if (memRefFullRank) {
        clang::QualType origTy = DT->getOriginalType();
        if (origTy->isConstantArrayType()) {
          SmallVector<int64_t, 4> shape;
          clang::QualType elemTy;
          getConstantArrayShapeAndElemType(origTy, shape, elemTy);

          return mlir::MemRefType::get(shape, getMLIRType(elemTy));
        }
      }

      // If -memref-fullrank is unset or it cannot be fulfilled.
      auto mt = mlirty.dyn_cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2[0] = -1;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    } else {
      return getMLIRType(DT->getAdjustedType(), implicitRef, allowMerge);
    }
    return mlirty;
  }
  if (auto CT = dyn_cast<clang::ComplexType>(qt)) {
    bool assumeRef = false;
    auto subType =
        getMLIRType(CT->getElementType(), &assumeRef, /*allowMerge*/ false);
    if (memRefABI && allowMerge) {
      assert(!assumeRef);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(2, subType);
    }
    mlir::Type types[2] = {subType, subType};
    return mlir::LLVM::LLVMStructType::getLiteral(module->getContext(), types);
  }
  if (auto RT = dyn_cast<clang::RecordType>(qt)) {
    if (RT->getDecl()->isInvalidDecl()) {
      RT->getDecl()->dump();
      RT->dump();
    }
    assert(!RT->getDecl()->isInvalidDecl());
    if (typeCache.find(RT) != typeCache.end())
      return typeCache[RT];
    llvm::Type *LT = CGM.getTypes().ConvertType(qt);
    if (!isa<llvm::StructType>(LT)) {
      qt->dump();
      llvm::errs() << "LT: " << *LT << "\n";
    }
    llvm::StructType *ST = cast<llvm::StructType>(LT);

    SmallPtrSet<llvm::Type *, 4> Seen;
    bool notAllSame = false;
    bool recursive = false;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      if (isRecursiveStruct(ST->getTypeAtIndex(i), ST, Seen)) {
        recursive = true;
      }
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
        notAllSame = true;
      }
    }

    const auto *RD = RT->getAsRecordDecl();
    if (mlirclang::isNamespaceSYCL(RD->getEnclosingNamespaceContext())) {
      const auto TypeName = RD->getName();
      if (TypeName == "range" || TypeName == "array" || TypeName == "id" ||
          TypeName == "accessor_common" || TypeName == "accessor" ||
          TypeName == "AccessorImplDevice" || TypeName == "item" ||
          TypeName == "ItemBase" || TypeName == "nd_item" ||
          TypeName == "group") {
        return getSYCLType(RT);
      }
      llvm::errs() << "Warning: SYCL type '" << ST->getName()
                   << "' has not been converted to SYCL MLIR\n";
    }

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (CodeGenUtils::isLLVMStructABI(RT->getDecl(), ST))
      return typeTranslator.translateType(anonymize(ST));

    /* TODO
    if (ST->getNumElements() == 1 && !recursive &&
        !RT->getDecl()->fields().empty() && ++RT->getDecl()->field_begin() ==
    RT->getDecl()->field_end()) { auto subT =
    getMLIRType((*RT->getDecl()->field_begin())->getType(), implicitRef,
    allowMerge); return subT;
    }
    */
    if (recursive)
      typeCache[RT] = LLVM::LLVMStructType::getIdentified(
          module->getContext(), ("polygeist@mlir@" + ST->getName()).str());

    SmallVector<mlir::Type, 4> types;

    bool innerLLVM = false;
    bool innerSYCL = false;
    if (CXRD) {
      for (auto f : CXRD->bases()) {
        bool subRef = false;
        auto ty = getMLIRType(f.getType(), &subRef, /*allowMerge*/ false);
        assert(!subRef);
        innerLLVM |= ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                            LLVM::LLVMArrayType>();
        types.push_back(ty);
      }
    }

    for (auto f : RT->getDecl()->fields()) {
      bool subRef = false;
      auto ty = getMLIRType(f->getType(), &subRef, /*allowMerge*/ false);
      assert(!subRef);
      innerLLVM |= ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                          LLVM::LLVMArrayType>();
      innerSYCL |=
          ty.isa<mlir::sycl::IDType, mlir::sycl::AccessorType,
                 mlir::sycl::RangeType, mlir::sycl::AccessorImplDeviceType,
                 mlir::sycl::ArrayType, mlir::sycl::ItemType,
                 mlir::sycl::ItemBaseType, mlir::sycl::NdItemType,
                 mlir::sycl::GroupType>();
      types.push_back(ty);
    }

    if (types.empty())
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return typeTranslator.translateType(anonymize(ST));

    if (recursive) {
      auto LR = typeCache[RT].setBody(types, /*isPacked*/ false);
      assert(LR.succeeded());
      return typeCache[RT];
    }

    if (!memRefABI || notAllSame || !allowMerge || innerLLVM || innerSYCL)
      return LLVM::LLVMStructType::getLiteral(module->getContext(), types);

    if (!types.size()) {
      RT->dump();
      llvm::errs() << "ST: " << *ST << "\n";
      llvm::errs() << "fields\n";
      for (auto f : RT->getDecl()->fields()) {
        llvm::errs() << " +++ ";
        f->getType()->dump();
        llvm::errs() << " @@@ " << *CGM.getTypes().ConvertType(f->getType())
                     << "\n";
      }
      llvm::errs() << "types\n";
      for (auto t : types)
        llvm::errs() << " --- " << t << "\n";
    }
    assert(types.size());
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get(types.size(), types[0]);
  }

  auto t = qt->getUnqualifiedDesugaredType();
  if (t->isVoidType()) {
    mlir::OpBuilder builder(module->getContext());
    return builder.getNoneType();
  }

  // if (auto AT = dyn_cast<clang::VariableArrayType>(t)) {
  //   return getMLIRType(AT->getElementType(), implicitRef, allowMerge);
  // }

  if (auto AT = dyn_cast<clang::ArrayType>(t)) {
    auto PTT = AT->getElementType()->getUnqualifiedDesugaredType();
    if (PTT->isCharType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = -1;
    if (auto CAT = dyn_cast<clang::ConstantArrayType>(AT))
      size = CAT->getSize().getZExtValue();
    if (memRefABI && subRef) {
      auto mt = ET.cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!memRefABI || !allowMerge ||
        ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
               LLVM::LLVMFunctionType, LLVM::LLVMStructType>())
      return LLVM::LLVMArrayType::get(ET, (size == -1) ? 0 : size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get(
        {size}, ET, {},
        CGM.getContext().getTargetAddressSpace(AT->getElementType()));
  }

  if (auto AT = dyn_cast<clang::VectorType>(t)) {
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = AT->getNumElements();
    if (memRefABI && subRef) {
      auto mt = ET.cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!memRefABI || !allowMerge ||
        ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
               LLVM::LLVMFunctionType, LLVM::LLVMStructType>())
      return LLVM::LLVMFixedVectorType::get(ET, size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (auto FT = dyn_cast<clang::FunctionProtoType>(t)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (RT.isa<mlir::NoneType>())
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    for (auto T : FT->getParamTypes()) {
      Args.push_back(getMLIRType(T));
    }
    return LLVM::LLVMFunctionType::get(RT, Args, FT->isVariadic());
  }
  if (auto FT = dyn_cast<clang::FunctionNoProtoType>(t)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (RT.isa<mlir::NoneType>())
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    return LLVM::LLVMFunctionType::get(RT, Args, /*isVariadic*/ true);
  }

  if (isa<clang::PointerType, clang::ReferenceType>(t)) {
    int64_t outer = -1;
    auto pointeeType = isa<clang::PointerType>(t)
                           ? cast<clang::PointerType>(t)->getPointeeType()
                           : cast<clang::ReferenceType>(t)->getPointeeType();
    auto PTT = pointeeType->getUnqualifiedDesugaredType();

    if (PTT->isCharType() || PTT->isVoidType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    bool subRef = false;
    auto subType = getMLIRType(pointeeType, &subRef, /*allowMerge*/ true);

    if (!memRefABI ||
        subType.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType,
                    LLVM::LLVMPointerType, LLVM::LLVMFunctionType>()) {
      // JLE_QUEL::THOUGHTS
      // When generating the sycl_halide_kernel, If a struct type contains
      // SYCL types, that means that this is the functor, and we can't create
      // a llvm pointer that contains custom aggregate types. We could create
      // a sycl::Functor type, that will help us get rid of those conditions.
      bool InnerSYCL = false;
      if (auto ST = subType.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        for (auto Element : ST.getBody()) {
          if (Element.isa<mlir::sycl::IDType, mlir::sycl::AccessorType,
                          mlir::sycl::RangeType,
                          mlir::sycl::AccessorImplDeviceType,
                          mlir::sycl::ArrayType, mlir::sycl::ItemType,
                          mlir::sycl::ItemBaseType, mlir::sycl::NdItemType,
                          mlir::sycl::GroupType>()) {
            InnerSYCL = true;
          }
        }
      }

      if (!InnerSYCL)
        return LLVM::LLVMPointerType::get(
            subType, CGM.getContext().getTargetAddressSpace(pointeeType));
    }

    if (isa<clang::ArrayType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        return subType;
      } else
        return LLVM::LLVMPointerType::get(subType);
    }

    if (isa<clang::VectorType>(PTT) || isa<clang::ComplexType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        auto mt = subType.cast<MemRefType>();
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
      } else
        return LLVM::LLVMPointerType::get(subType);
    }

    if (isa<clang::RecordType>(PTT))
      if (subRef) {
        auto mt = subType.cast<MemRefType>();
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
      }

    assert(!subRef);
    return mlir::MemRefType::get(
        {outer}, subType, {},
        CGM.getContext().getTargetAddressSpace(pointeeType));
  }

  if (t->isBuiltinType() || isa<clang::EnumType>(t)) {
    if (t->isBooleanType()) {
      OpBuilder builder(module->getContext());
      return builder.getIntegerType(8);
    }
    llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
    mlir::OpBuilder builder(module->getContext());
    if (T->isVoidTy()) {
      return builder.getNoneType();
    }
    if (T->isFloatTy()) {
      return builder.getF32Type();
    }
    if (T->isDoubleTy()) {
      return builder.getF64Type();
    }
    if (T->isX86_FP80Ty())
      return builder.getF80Type();
    if (T->isFP128Ty())
      return builder.getF128Type();

    if (auto IT = dyn_cast<llvm::IntegerType>(T)) {
      return builder.getIntegerType(IT->getBitWidth());
    }
  }
  qt->dump();
  assert(0 && "unhandled type");
}

mlir::Type MLIRASTConsumer::getSYCLType(const clang::RecordType *RT) {
  const auto *RD = RT->getAsRecordDecl();
  llvm::SmallVector<mlir::Type, 4> Body;

  for (const auto *Field : RD->fields()) {
    Body.push_back(getMLIRType(Field->getType()));
  }

  if (const auto *CTS =
          llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(RD)) {
    if (CTS->getName() == "range") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::RangeType::get(module->getContext(), Dim);
    }
    if (CTS->getName() == "array") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::ArrayType::get(module->getContext(), Dim, Body);
    }
    if (CTS->getName() == "id") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::IDType::get(module->getContext(), Dim);
    }
    if (CTS->getName() == "accessor_common") {
      return mlir::sycl::AccessorCommonType::get(module->getContext());
    }
    if (CTS->getName() == "accessor") {
      const auto Type = getMLIRType(CTS->getTemplateArgs().get(0).getAsType());
      const auto Dim =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      const auto MemAccessMode = static_cast<mlir::sycl::MemoryAccessMode>(
          CTS->getTemplateArgs().get(2).getAsIntegral().getExtValue());
      const auto MemTargetMode = static_cast<mlir::sycl::MemoryTargetMode>(
          CTS->getTemplateArgs().get(3).getAsIntegral().getExtValue());
      return mlir::sycl::AccessorType::get(module->getContext(), Type, Dim,
                                           MemAccessMode, MemTargetMode, Body);
    }
    if (CTS->getName() == "AccessorImplDevice") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::AccessorImplDeviceType::get(module->getContext(), Dim,
                                                     Body);
    }
    if (CTS->getName() == "item") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      const auto Offset =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::ItemType::get(module->getContext(), Dim, Offset, Body);
    }
    if (CTS->getName() == "ItemBase") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      const auto Offset =
          CTS->getTemplateArgs().get(1).getAsIntegral().getExtValue();
      return mlir::sycl::ItemBaseType::get(module->getContext(), Dim, Offset,
                                           Body);
    }
    if (CTS->getName() == "nd_item") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::NdItemType::get(module->getContext(), Dim, Body);
    }
    if (CTS->getName() == "group") {
      const auto Dim =
          CTS->getTemplateArgs().get(0).getAsIntegral().getExtValue();
      return mlir::sycl::GroupType::get(module->getContext(), Dim, Body);
    }
  }

  llvm_unreachable("SYCL type not handle (yet)");
}

llvm::Type *MLIRASTConsumer::getLLVMType(clang::QualType t) {
  if (t->isVoidType()) {
    return llvm::Type::getVoidTy(llvmMod.getContext());
  }
  llvm::Type *T = CGM.getTypes().ConvertType(t);
  return T;
}

#include "clang/Frontend/FrontendAction.h"
#include "llvm/Support/Host.h"

class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> emitIfFound;
  std::set<std::pair<FunctionContext, std::string>> done;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> globals;
  std::map<std::string, mlir::func::FuncOp> functions;
  std::map<std::string, mlir::FunctionOpInterface> deviceFunctions;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  std::string moduleId;
  MLIRAction(std::string fn, mlir::OwningOpRef<mlir::ModuleOp> &module,
             std::string moduleId)
      : module(module), moduleId(moduleId) {
    emitIfFound.insert(fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new MLIRASTConsumer(
        emitIfFound, done, llvmStringGlobals, globals, functions,
        deviceFunctions, llvmGlobals, llvmFunctions, CI.getPreprocessor(),
        CI.getASTContext(), module, CI.getSourceManager(), CI.getCodeGenOpts(),
        moduleId));
  }
};

mlir::FunctionOpInterface
MLIRScanner::EmitDirectCallee(const FunctionDecl *FD, FunctionContext Context) {
  FunctionToEmit FTE(*FD, Context);
  return Glob.GetOrCreateMLIRFunction(FTE, false /* IsThunk */,
                                      true /* ShouldEmit */);
}

mlir::Location MLIRScanner::getMLIRLocation(clang::SourceLocation loc) {
  return Glob.getMLIRLocation(loc);
}

mlir::Type MLIRScanner::getMLIRType(clang::QualType t) {
  return Glob.getMLIRType(t);
}

llvm::Type *MLIRScanner::getLLVMType(clang::QualType t) {
  return Glob.getLLVMType(t);
}

mlir::Value MLIRScanner::getTypeSize(clang::QualType t) {
  // llvm::Type *T = Glob.getCGM().getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getMLIRType(t, &isArray);
  if (isArray) {
    auto MT = innerTy.cast<MemRefType>();
    size_t num = 1;
    for (auto n : MT.getShape()) {
      assert(n > 0);
      num *= n;
    }
    return builder.create<arith::MulIOp>(
        loc,
        builder.create<polygeist::TypeSizeOp>(
            loc, builder.getIndexType(),
            mlir::TypeAttr::get(MT.getElementType())),
        builder.create<arith::ConstantIndexOp>(loc, num));
  }
  assert(!isArray);
  return builder.create<polygeist::TypeSizeOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

mlir::Value MLIRScanner::getTypeAlign(clang::QualType t) {
  // llvm::Type *T = Glob.getCGM().getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getMLIRType(t, &isArray);
  assert(!isArray);
  return builder.create<polygeist::TypeAlignOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

std::string MLIRScanner::getMangledFuncName(const FunctionDecl &FD,
                                            CodeGen::CodeGenModule &CGM) {
  if (auto CC = dyn_cast<CXXConstructorDecl>(&FD))
    return CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  if (auto CC = dyn_cast<CXXDestructorDecl>(&FD))
    return CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();

  return CGM.getMangledName(&FD).str();
}

#include "clang/Frontend/TextDiagnosticBuffer.h"

static bool parseMLIR(const char *Argv0, std::vector<std::string> filenames,
                      std::string fn, std::vector<std::string> includeDirs,
                      std::vector<std::string> defines,
                      mlir::OwningOpRef<mlir::ModuleOp> &module,
                      llvm::Triple &triple, llvm::DataLayout &DL,
                      std::vector<std::string> InputCommandArgs) {

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  bool Success;
  //{
  const char *binary = Argv0; // CudaLower ? "clang++" : "clang";
  const std::unique_ptr<Driver> driver(
      new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  std::vector<const char *> Argv;
  Argv.push_back(binary);
  for (auto a : filenames) {
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (TargetTripleOpt != "") {
    char *chars = (char *)malloc(TargetTripleOpt.length() + 1);
    memcpy(chars, TargetTripleOpt.data(), TargetTripleOpt.length());
    chars[TargetTripleOpt.length()] = 0;
    Argv.push_back("-target");
    Argv.push_back(chars);
  }
  if (McpuOpt != "") {
    auto a = "-mcpu=" + McpuOpt;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (Standard != "") {
    auto a = "-std=" + Standard;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    char *chars = (char *)malloc(ResourceDir.length() + 1);
    memcpy(chars, ResourceDir.data(), ResourceDir.length());
    chars[ResourceDir.length()] = 0;
    Argv.push_back(chars);
  }
  if (SysRoot != "") {
    Argv.push_back("--sysroot");
    char *chars = (char *)malloc(SysRoot.length() + 1);
    memcpy(chars, SysRoot.data(), SysRoot.length());
    chars[SysRoot.length()] = 0;
    Argv.push_back(chars);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (NoCUDAInc) {
    Argv.push_back("-nocudainc");
  }
  if (NoCUDALib) {
    Argv.push_back("-nocudalib");
  }
  if (CUDAGPUArch != "") {
    auto a = "--cuda-gpu-arch=" + CUDAGPUArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (CUDAPath != "") {
    auto a = "--cuda-path=" + CUDAPath;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (MArch != "") {
    auto a = "-march=" + MArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : includeDirs) {
    Argv.push_back("-I");
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : defines) {
    char *chars = (char *)malloc(a.length() + 3);
    chars[0] = '-';
    chars[1] = 'D';
    memcpy(chars + 2, a.data(), a.length());
    chars[2 + a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : Includes) {
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back("-include");
    Argv.push_back(chars);
  }

  Argv.push_back("-emit-ast");

  llvm::SmallVector<const ArgStringList *, 4> CommandList;
  ArgStringList InputCommandArgList;

  std::unique_ptr<Compilation> compilation;

  if (InputCommandArgs.empty()) {
    compilation.reset(std::move(
        driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv))));

    JobList &Jobs = compilation->getJobs();
    if (Jobs.size() < 1)
      return false;
    for (auto &job : Jobs) {
      Command *cmd = cast<Command>(&job);
      if (strcmp(cmd->getCreator().getName(), "clang"))
        return false;
      CommandList.push_back(&cmd->getArguments());
    }
  } else {
    for (std::string &s : InputCommandArgs) {
      InputCommandArgList.push_back(s.c_str());
    }
    CommandList.push_back(&InputCommandArgList);
  }

  MLIRAction Act(fn, module,
                 filenames.size() == 1 ? filenames[0] : "LLVMDialectModule");

  for (const ArgStringList *args : CommandList) {
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.size() == 0)
      Clang->getHeaderSearchOpts().ResourceDir =
          CompilerInvocation::GetResourcesPath(Argv0, GetExecutablePathVP);

    //}
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();
    if (!Clang->hasDiagnostics())
      return false;

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
    if (!Success)
      return false;

    // Create and execute the frontend action.

    // Create the target instance.
    Clang->setTarget(TargetInfo::CreateTargetInfo(
        Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
    if (!Clang->hasTarget())
      return false;

    // Create TargetInfo for the other side of CUDA and OpenMP compilation.
    if ((Clang->getLangOpts().CUDA || Clang->getLangOpts().OpenMPIsDevice ||
         Clang->getLangOpts().SYCLIsDevice) &&
        !Clang->getFrontendOpts().AuxTriple.empty()) {
      auto TO = std::make_shared<clang::TargetOptions>();
      TO->Triple = llvm::Triple::normalize(Clang->getFrontendOpts().AuxTriple);
      TO->HostTriple = Clang->getTarget().getTriple().str();
      Clang->setAuxTarget(
          TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), TO));
    }

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts());

    // Adjust target options based on codegen options.
    Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(),
                                           Clang->getTargetOpts());

    llvm::Triple jobTriple = Clang->getTarget().getTriple();
    if (triple.str() == "" || !jobTriple.isNVPTX()) {
      triple = jobTriple;
      module.get()->setAttr(
          LLVM::LLVMDialect::getTargetTripleAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      DL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      module.get()->setAttr(
          LLVM::LLVMDialect::getDataLayoutAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getDataLayoutString()));

      module.get()->setAttr(("dlti." + DataLayoutSpecAttr::kAttrKeyword).str(),
                            translateDataLayout(DL, module->getContext()));
    }

    for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
      // Reset the ID tables if we are reusing the SourceManager and parsing
      // regular files.
      if (Clang->hasSourceManager() && !Act.isModelParsingAction())
        Clang->getSourceManager().clearIDTables();
      if (Act.BeginSourceFile(*Clang, FIF)) {

        llvm::Error err = Act.Execute();
        if (err) {
          llvm::errs() << "saw error: " << err << "\n";
          return false;
        }
        assert(Clang->hasSourceManager());

        Act.EndSourceFile();
      }
    }
  }
  return true;
}
