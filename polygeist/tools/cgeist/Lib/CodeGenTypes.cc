//===--- CodeGenTypes.cc - Type translation for MLIR CodeGen -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> MLIR type lowering.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTypes.h"
#include "TypeUtils.h"
#include "utils.h"

#include "clang/../../lib/CodeGen/CGOpenCLRuntime.h"
#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/CodeGen/CGFunctionInfo.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

#include "llvm/IR/Assumptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cgeist"

using namespace clang;
using namespace mlir;
using namespace llvm;

static cl::opt<bool>
    MemRefFullRank("memref-fullrank", cl::init(false),
                   cl::desc("Get the full rank of the memref."));

static cl::opt<bool> MemRefABI("memref-abi", cl::init(true),
                               cl::desc("Use memrefs when possible"));

static cl::opt<bool>
    CombinedStructABI("struct-abi", cl::init(true),
                      cl::desc("Use literal LLVM ABI for structs"));

static cl::opt<bool>
    AllowUndefinedSYCLTypes("allow-undefined-sycl-types", cl::init(false),
                            cl::desc("Whether to allow types in the sycl "
                                     "namespace and not in the SYCL dialect"));

/******************************************************************************/
/*            Flags affecting code generation of function types.              */
/******************************************************************************/

// Note: cgeist does not allow flattening struct function parameters. Need to
// revisit.
constexpr bool AllowStructFlattening = false;

// Note: cgeist does not allow returning a struct via the parameter list. Need
// to revisit.
constexpr bool AllowSRet = false;

// Note: cgesit does not allow returning 'inalloca'. Need to revisit.
constexpr bool AllowInAllocaRet = false;

/******************************************************************************/
/*                            Helper Functions                                */
/******************************************************************************/

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS, clang::CodeGen::ABIArgInfo::Kind &ArgInfo) {
  OS << "ABIArgInfo::";
  switch (ArgInfo) {
  case clang::CodeGen::ABIArgInfo::Direct:
    return OS << "Direct";
  case clang::CodeGen::ABIArgInfo::Extend:
    return OS << "Extend";
  case clang::CodeGen::ABIArgInfo::Indirect:
    return OS << "Indirect";
  case clang::CodeGen::ABIArgInfo::IndirectAliased:
    return OS << "IndirectAliased";
  case clang::CodeGen::ABIArgInfo::Ignore:
    return OS << "Ignore";
  case clang::CodeGen::ABIArgInfo::Expand:
    return OS << "Expand";
  case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
    return OS << "CoerceAndExpand";
  case clang::CodeGen::ABIArgInfo::InAlloca:
    return OS << "InAlloca";
  }
  llvm_unreachable("Invalid ABI kind");
  return OS;
}

/// Iteratively get the size of each dim of the given ConstantArrayType inst.
static void
getConstantArrayShapeAndElemType(const clang::QualType &Ty,
                                 llvm::SmallVectorImpl<int64_t> &Shape,
                                 clang::QualType &ElemTy) {
  Shape.clear();

  clang::QualType CurTy = Ty;
  while (CurTy->isConstantArrayType()) {
    const auto *CstArrTy = cast<clang::ConstantArrayType>(CurTy);
    Shape.push_back(CstArrTy->getSize().getSExtValue());
    CurTy = CstArrTy->getElementType();
  }

  ElemTy = CurTy;
}

static constexpr int AllocSizeNumElemsNotPresent = -1;

static uint64_t packAllocSizeArgs(unsigned ElemSizeArg,
                                  const Optional<int> &NumElemsArg) {
  assert((!NumElemsArg || *NumElemsArg != AllocSizeNumElemsNotPresent) &&
         "Attempting to pack a reserved value");

  return uint64_t(ElemSizeArg) << 32 |
         NumElemsArg.value_or(AllocSizeNumElemsNotPresent);
}

static void
addAttributesFromFunctionProtoType(mlirclang::AttrBuilder &FuncAttrs,
                                   const clang::FunctionProtoType *FPT) {
  if (!FPT)
    return;

  if (!isUnresolvedExceptionSpec(FPT->getExceptionSpecType()) &&
      FPT->isNothrow())
    FuncAttrs.addPassThroughAttribute(llvm::Attribute::NoUnwind);
}

static void addAttributesFromAssumes(mlirclang::AttrBuilder &FuncAttrs,
                                     const Decl *Callee) {
  if (!Callee)
    return;

  SmallVector<StringRef, 4> Attrs;
  for (const AssumptionAttr *AA : Callee->specific_attrs<AssumptionAttr>())
    AA->getAssumption().split(Attrs, ",");

  MLIRContext *Ctx = &FuncAttrs.getContext();
  SmallVector<mlir::Attribute, 4> StrAttrs;
  for (StringRef AttrName : Attrs)
    StrAttrs.push_back(StringAttr::get(Ctx, AttrName));

  FuncAttrs.addPassThroughAttribute(llvm::AssumptionAttrKey,
                                    mlir::ArrayAttr::get(Ctx, StrAttrs));
}

static void addNoBuiltinAttributes(mlirclang::AttrBuilder &FuncAttrs,
                                   const LangOptions &LangOpts,
                                   const NoBuiltinAttr *NBA = nullptr) {
  auto AddNoBuiltinAttr = [&FuncAttrs](StringRef BuiltinName) {
    SmallString<32> AttributeName;
    AttributeName += "no-builtin-";
    AttributeName += BuiltinName;
    FuncAttrs.addPassThroughAttribute(AttributeName, mlir::UnitAttr());
  };

  // First, handle the language options passed through -fno-builtin.
  if (LangOpts.NoBuiltin) {
    // -fno-builtin disables them all.
    FuncAttrs.addPassThroughAttribute("no-builtins", mlir::UnitAttr());
    return;
  }

  // Then, add attributes for builtins specified through -fno-builtin-<name>.
  llvm::for_each(LangOpts.NoBuiltinFuncs, AddNoBuiltinAttr);

  // Now, let's check the __attribute__((no_builtin("...")) attribute added to
  // the source.
  if (!NBA)
    return;

  // If there is a wildcard in the builtin names specified through the
  // attribute, disable them all.
  if (llvm::is_contained(NBA->builtinNames(), "*")) {
    FuncAttrs.addPassThroughAttribute("no-builtins", mlir::UnitAttr());
    return;
  }

  // And last, add the rest of the builtin names.
  llvm::for_each(NBA->builtinNames(), AddNoBuiltinAttr);
}

static bool determineNoUndef(QualType QT, clang::CodeGen::CodeGenTypes &Types,
                             const llvm::DataLayout &DL,
                             const clang::CodeGen::ABIArgInfo &AI,
                             bool CheckCoerce = true) {
  llvm::Type *Ty = Types.ConvertTypeForMem(QT);
  if (AI.getKind() == clang::CodeGen::ABIArgInfo::Indirect)
    return true;
  if (AI.getKind() == clang::CodeGen::ABIArgInfo::Extend)
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
  if (QT->isBitIntType())
    return true;
  if (QT->isReferenceType())
    return true;
  if (QT->isNullPtrType())
    return false;
  if (QT->isMemberPointerType())
    // TODO: Some member pointers are `noundef`, but it depends on the ABI. For
    // now, never mark them.
    return false;
  if (QT->isScalarType()) {
    if (const auto *Complex = dyn_cast<clang::ComplexType>(QT))
      return determineNoUndef(Complex->getElementType(), Types, DL, AI, false);
    return true;
  }
  if (const auto *Vector = dyn_cast<clang::VectorType>(QT))
    return determineNoUndef(Vector->getElementType(), Types, DL, AI, false);
  if (const auto *Matrix = dyn_cast<clang::MatrixType>(QT))
    return determineNoUndef(Matrix->getElementType(), Types, DL, AI, false);
  if (const auto *Array = dyn_cast<clang::ArrayType>(QT))
    return determineNoUndef(Array->getElementType(), Types, DL, AI, false);

  // TODO: Some structs may be `noundef`, in specific situations.
  return false;
}

static bool isSingleFieldUnion(const clang::RecordDecl *RD) {
  assert(RD->isUnion() && "Expecting union input");
  return std::next(RD->field_begin()) == RD->field_end();
}

namespace mlirclang {
namespace CodeGen {

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

  llvm::SmallVector<IRArgs, 8> ArgInfo;

public:
  ClangToLLVMArgMapping(const clang::ASTContext &Context,
                        const clang::CodeGen::CGFunctionInfo &FI,
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
  void construct(const clang::ASTContext &Context,
                 const clang::CodeGen::CGFunctionInfo &FI,
                 bool OnlyRequiredArgs);
};

void ClangToLLVMArgMapping::construct(const clang::ASTContext &Context,
                                      const clang::CodeGen::CGFunctionInfo &FI,
                                      bool OnlyRequiredArgs) {
  unsigned IRArgNo = 0;
  bool SwapThisWithSRet = false;
  const clang::CodeGen::ABIArgInfo &RetAI = FI.getReturnInfo();

  if (AllowSRet && RetAI.getKind() == clang::CodeGen::ABIArgInfo::Indirect) {
    SwapThisWithSRet = RetAI.isSRetAfterThis();
    SRetArgNo = SwapThisWithSRet ? 1 : IRArgNo++;
  }

  unsigned ArgNo = 0;
  unsigned NumArgs = OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
  for (clang::CodeGen::CGFunctionInfo::const_arg_iterator I = FI.arg_begin();
       ArgNo < NumArgs; ++I, ++ArgNo) {
    assert(I != FI.arg_end());
    const clang::CodeGen::ABIArgInfo &AI = I->info;
    // Collect data about IR arguments corresponding to Clang argument ArgNo.
    auto &IRArgs = ArgInfo[ArgNo];

    if (AI.getPaddingType())
      IRArgs.PaddingArgIndex = IRArgNo++;

    switch (AI.getKind()) {
    case clang::CodeGen::ABIArgInfo::Extend:
    case clang::CodeGen::ABIArgInfo::Direct: {
      // FIXME: handle sseregparm someday...
      llvm::StructType *STy = dyn_cast<llvm::StructType>(AI.getCoerceToType());

      CGEIST_WARNING({
        if (AI.isDirect() && AI.getCanBeFlattened() && STy)
          llvm::WithColor::warning()
              << "struct should be flattened but MLIR codegen "
                 "cannot yet handle it. Needs to be fixed.\n";
      });

      if (AllowStructFlattening && AI.isDirect() && AI.getCanBeFlattened() &&
          STy) {
        IRArgs.NumberOfArgs = STy->getNumElements();
      } else {
        IRArgs.NumberOfArgs = 1;
      }
      break;
    }
    case clang::CodeGen::ABIArgInfo::Indirect:
    case clang::CodeGen::ABIArgInfo::IndirectAliased:
      IRArgs.NumberOfArgs = 1;
      break;
    case clang::CodeGen::ABIArgInfo::Ignore:
    case clang::CodeGen::ABIArgInfo::InAlloca:
      // ignore and inalloca doesn't have matching LLVM parameters.
      IRArgs.NumberOfArgs = 0;
      break;
    case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
    case clang::CodeGen::ABIArgInfo::Expand:
      llvm_unreachable("not implemented");
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

  if (AllowInAllocaRet && FI.usesInAlloca())
    InallocaArgNo = IRArgNo++;

  TotalIRArgs = IRArgNo;
}
} // namespace

CodeGenTypes::CodeGenTypes(clang::CodeGen::CodeGenModule &CGM,
                           mlir::OwningOpRef<mlir::ModuleOp> &Module)
    : CGM(CGM), Context(CGM.getContext()), TheModule(Module),
      TheCXXABI(CGM.getCXXABI()) {}

const CodeGenOptions &CodeGenTypes::getCodeGenOpts() const {
  return CGM.getCodeGenOpts();
}

mlir::FunctionType
CodeGenTypes::getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                              const clang::FunctionDecl &FD) {
  LLVM_DEBUG(llvm::dbgs() << "\n-- Entering getFunctionType --\n");

  const bool IsMethodDecl = isa<CXXMethodDecl>(FD);
  const bool IsMethodInstance =
      IsMethodDecl && cast<CXXMethodDecl>(FD).isInstance();
  bool IsArrayReturn = false;
  getMLIRType(FD.getReturnType(), &IsArrayReturn);

  LLVM_DEBUG({
    if (IsMethodInstance)
      llvm::dbgs() << "IsMethodInstance = true\n";
    if (IsArrayReturn)
      llvm::dbgs() << "IsArrayReturn = true\n";
  });

  // This lambda function returns the declared type of a function parameter when
  // given the index of the parameter and the decayed type of the parameter.
  auto GetDeclArgTy = [&](int32_t ArgNo, QualType ABIArgTy) {
    if (IsMethodInstance) // account for the fact the 'this' type is not
      ArgNo--;            // present in the function declaration.
    const QualType DeclArgTy =
        (ArgNo == -1) ? ABIArgTy : FD.getParamDecl(ArgNo)->getType();

    LLVM_DEBUG({
      llvm::dbgs() << "ABIArgTy: ";
      ABIArgTy.dump();
      llvm::dbgs() << "DeclArgTy: ";
      DeclArgTy.dump();
    });

    return DeclArgTy;
  };

  // This lambda function returns the MLIR type corresponding to the given clang
  // type.
  auto GetMLIRArgType = [this](QualType QT) {
    bool IsRef = false;
    mlir::Type ArgTy = getMLIRType(QT, &IsRef);
    if (IsRef)
      ArgTy = getMLIRType(CGM.getContext().getLValueReferenceType(QT));
    return ArgTy;
  };

  //
  // Compute the type of the function return value.
  //
  const clang::CodeGen::ABIArgInfo &RetAI = FI.getReturnInfo();
  ClangToLLVMArgMapping IRFunctionArgs(CGM.getContext(), FI, true);
  mlir::OpBuilder Builder(TheModule->getContext());
  LLVM_DEBUG(llvm::dbgs() << "Processing return value\n");

  mlir::Type ResultType;
  clang::CodeGen::ABIArgInfo::Kind Kind = RetAI.getKind();
  LLVM_DEBUG(llvm::dbgs() << "RetInfo: " << Kind << "\n");
  switch (Kind) {
  case clang::CodeGen::ABIArgInfo::Expand:
  case clang::CodeGen::ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");

  case clang::CodeGen::ABIArgInfo::Extend:
  case clang::CodeGen::ABIArgInfo::Direct:
    ResultType =
        IsArrayReturn ? Builder.getNoneType() : getMLIRType(FI.getReturnType());
    break;

  case clang::CodeGen::ABIArgInfo::InAlloca:
    if (RetAI.getInAllocaSRet()) {
      // sret things on win32 aren't void, they return the sret pointer.
      QualType Ret = FI.getReturnType();
      mlir::Type Ty = getMLIRType(Ret);
      unsigned AddressSpace =
          CGM.getContext().getTargetAddressSpace(Ret.getAddressSpace());
      ResultType = getPointerOrMemRefType(Ty, AddressSpace);
    } else {
      ResultType = Builder.getNoneType();
    }
    break;

  case clang::CodeGen::ABIArgInfo::Indirect:
    if (!AllowSRet) {
      // HACK: remove once we can handle function returning a struct.
      CGEIST_WARNING(llvm::WithColor::warning()
                     << "function should return its value indirectly (as "
                        "an extra reference parameter). This is not yet "
                        "handled by the MLIR codegen\n");
      QualType Ret = FI.getReturnType();
      ResultType = getMLIRType(Ret);
      break;
    }
  case clang::CodeGen::ABIArgInfo::Ignore:
    ResultType = Builder.getNoneType();
    break;

  case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
    llvm_unreachable("not implemented");
    break;
  }

  //
  // Compute the types of the function parameters.
  //
  const unsigned NumArgs = IsArrayReturn ? IRFunctionArgs.totalIRArgs() + 1
                                         : IRFunctionArgs.totalIRArgs();
  SmallVector<mlir::Type, 8> ArgTypes(NumArgs);
  LLVM_DEBUG(llvm::dbgs() << "NumArgs = " << NumArgs << "\n");

  // Add type for sret argument.
  if (AllowSRet && IRFunctionArgs.hasSRetArg()) {
    llvm_unreachable("not implemented");
    QualType Ret = FI.getReturnType();
    mlir::Type Ty = getMLIRType(Ret);
    unsigned AddressSpace =
        CGM.getContext().getTargetAddressSpace(Ret.getAddressSpace());
    ArgTypes[IRFunctionArgs.getSRetArgNo()] =
        getPointerOrMemRefType(Ty, AddressSpace);
  }

  // Add type for inalloca argument.
  if (AllowInAllocaRet && IRFunctionArgs.hasInallocaArg()) {
    llvm_unreachable("not implemented");
    auto *ArgStruct = FI.getArgStruct();
    assert(ArgStruct);
    // ArgTypes[IRFunctionArgs.getInallocaArgNo()] = ArgStruct->getPointerTo();
  }

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  clang::CodeGen::CGFunctionInfo::const_arg_iterator
      It = FI.arg_begin(),
      Ie = It + FI.getNumRequiredArgs();
  for (; It != Ie; ++It, ++ArgNo) {
    // Note: 'ArgTy' is the type of the parameter after it as been decayed to
    // abide to the ABI rules.
    const QualType &ArgTy = It->type;
    const clang::CodeGen::ABIArgInfo &ArgInfo = It->info;

    // TODO: Currently cgeist does not handle inserting paddings. Need to
    // revisit.
    bool InsertPadding = false;

    // Insert a padding type to ensure proper alignment.
    if (InsertPadding && IRFunctionArgs.hasPaddingArg(ArgNo)) {
      llvm_unreachable("not implemented");
      // ArgTypes[IRFunctionArgs.getPaddingArgNo(ArgNo)] =
      //    ArgInfo.getPaddingType();
    }

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    LLVM_DEBUG({
      llvm::dbgs() << "\nProcessing Arg " << ArgNo
                   << ", FirstIRArg = " << FirstIRArg
                   << ", NumIRArgs = " << NumIRArgs << "\n";
    });

    // Note: 'DeclArgTy' is the original type of the parameter in the
    // function declaration. In order to avoid premature loss of information
    // (e.g. extent of array dimensions) we want to use the original type.
    const QualType DeclArgTy = GetDeclArgTy(ArgNo, ArgTy);

    clang::CodeGen::ABIArgInfo::Kind Kind = ArgInfo.getKind();
    LLVM_DEBUG(llvm::dbgs() << "ArgInfo: " << Kind << "\n");

    switch (Kind) {
    case clang::CodeGen::ABIArgInfo::Ignore:
    case clang::CodeGen::ABIArgInfo::InAlloca:
      assert(NumIRArgs == 0);
      break;
    case clang::CodeGen::ABIArgInfo::Indirect: {
      assert(NumIRArgs == 1);
      // indirect arguments are always on the stack, which is alloca addr space.
      mlir::Type MLIRArgTy = GetMLIRArgType(DeclArgTy);
      ArgTypes[FirstIRArg] = getPointerOrMemRefType(
          MLIRArgTy, CGM.getDataLayout().getAllocaAddrSpace());
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "mlir type: " << ArgTypes[FirstIRArg] << "\n");
      break;
    }
    case clang::CodeGen::ABIArgInfo::IndirectAliased: {
      assert(NumIRArgs == 1);
      mlir::Type MLIRArgTy = GetMLIRArgType(DeclArgTy);
      ArgTypes[FirstIRArg] =
          getPointerOrMemRefType(MLIRArgTy, ArgInfo.getIndirectAddrSpace());
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "mlir type: " << ArgTypes[FirstIRArg] << "\n");
      break;
    }
    case clang::CodeGen::ABIArgInfo::Extend:
    case clang::CodeGen::ABIArgInfo::Direct: {
      mlir::Type MLIRArgTy = GetMLIRArgType(DeclArgTy);

      // Fast-isel and the optimizer generally like
      // scalar values better than FCAs, so we flatten them if this is safe to
      // do for this argument.
      auto ST = MLIRArgTy.dyn_cast<mlir::LLVM::LLVMStructType>();

      CGEIST_WARNING({
        if (ST && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened())
          llvm::WithColor::warning()
              << "struct should be flattened but MLIR codegen "
                 "cannot yet handle it. Needs to be fixed.\n";
      });

      if (AllowStructFlattening && ST && ArgInfo.isDirect() &&
          ArgInfo.getCanBeFlattened()) {
        assert(NumIRArgs == ST.getBody().size());
        for (unsigned I = 0, E = ST.getBody().size(); I != E; ++I)
          ArgTypes[FirstIRArg + I] = ST.getBody()[I];
      } else {
        assert(NumIRArgs == 1);
        ArgTypes[FirstIRArg] = MLIRArgTy;
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "mlir type: " << ArgTypes[FirstIRArg] << "\n");
      }
      break;
    }
    case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
    case clang::CodeGen::ABIArgInfo::Expand:
      llvm_unreachable("not implemented");
    }
  }

  // We return arrays via the parameter list to mirror cgeist special handling
  // for functions returning an array.
  // Note: this is not conforming to the ABI and should be fixed.
  if (IsArrayReturn) {
    auto MLIRType = getMLIRType(
        CGM.getContext().getLValueReferenceType(FD.getReturnType()));
    assert(MLIRType.isa<MemRefType>() &&
           MLIRType.cast<MemRefType>().getShape().size() == 2);
    ArgTypes[NumArgs - 1] = MLIRType;
    LLVM_DEBUG({
      llvm::dbgs() << "Added parameter for array return\n";
      llvm::dbgs().indent(2) << "mlir type: " << MLIRType << "\n";
    });
  }

  SmallVector<mlir::Type, 2> ResultTypes;
  if (!ResultType.isa<mlir::NoneType>())
    ResultTypes.push_back(ResultType);

  assert(llvm::all_of(ArgTypes, [](mlir::Type t) { return t; }) &&
         "ArgTypes should not contain a null type");
  assert(llvm::all_of(ResultTypes, [](mlir::Type t) { return t; }) &&
         "ResultTypes should not contain a null type");

  return Builder.getFunctionType(ArgTypes, ResultTypes);
}

void CodeGenTypes::constructAttributeList(
    StringRef Name, const clang::CodeGen::CGFunctionInfo &FI,
    clang::CodeGen::CGCalleeInfo CalleeInfo, mlirclang::AttributeList &AttrList,
    unsigned &CallingConv, bool AttrOnCallSite, bool IsThunk) {
  MLIRContext *Ctx = TheModule->getContext();
  mlirclang::AttrBuilder FuncAttrsBuilder(*Ctx);
  mlirclang::AttrBuilder RetAttrsBuilder(*Ctx);

  CallingConv = FI.getEffectiveCallingConvention();
  if (FI.isNoReturn())
    FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoReturn);
  if (FI.isCmseNSCall())
    FuncAttrsBuilder.addPassThroughAttribute("cmse_nonsecure_call",
                                             UnitAttr::get(Ctx));

  // Collect function IR attributes from the callee prototype if we have one.
  addAttributesFromFunctionProtoType(FuncAttrsBuilder,
                                     CalleeInfo.getCalleeFunctionProtoType());

  const Decl *TargetDecl = CalleeInfo.getCalleeDecl().getDecl();

  // Attach assumption attributes to the declaration. If this is a call
  // site, attach assumptions from the caller to the call as well.
  addAttributesFromAssumes(FuncAttrsBuilder, TargetDecl);

  bool HasOptnone = false;
  // The NoBuiltinAttr attached to the target FunctionDecl.
  const NoBuiltinAttr *NBA = nullptr;

  // Some ABIs may result in additional accesses to arguments that may otherwise
  // not be present.
  auto AddPotentialArgAccess = [&]() {
    llvm::Optional<mlir::NamedAttribute> A =
        FuncAttrsBuilder.getAttribute(llvm::Attribute::Memory);
    if (A) {
      IntegerAttr AA = A->getValue().cast<IntegerAttr>();
      auto ME = llvm::MemoryEffects::createFromIntValue(AA.getInt()) |
                llvm::MemoryEffects::argMemOnly();
      FuncAttrsBuilder.addAttribute(llvm::Attribute::Memory, ME.toIntValue());
    }
  };

  // Collect function IR attributes based on declaration-specific
  // information.
  if (TargetDecl) {
    if (TargetDecl->hasAttr<ReturnsTwiceAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::ReturnsTwice);
    if (TargetDecl->hasAttr<NoThrowAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoUnwind);
    if (TargetDecl->hasAttr<NoReturnAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoReturn);
    if (TargetDecl->hasAttr<ColdAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::Cold);
    if (TargetDecl->hasAttr<HotAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::Hot);
    if (TargetDecl->hasAttr<NoDuplicateAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoDuplicate);
    if (TargetDecl->hasAttr<ConvergentAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::Convergent);

    if (const FunctionDecl *Fn = dyn_cast<FunctionDecl>(TargetDecl)) {
      addAttributesFromFunctionProtoType(
          FuncAttrsBuilder, Fn->getType()->getAs<FunctionProtoType>());
      if (AttrOnCallSite && Fn->isReplaceableGlobalAllocationFunction()) {
        // A sane operator new returns a non-aliasing pointer.
        auto Kind = Fn->getDeclName().getCXXOverloadedOperator();
        if (CGM.getCodeGenOpts().AssumeSaneOperatorNew &&
            (Kind == OO_New || Kind == OO_Array_New))
          RetAttrsBuilder.addAttribute(llvm::Attribute::NoAlias);
      }
      const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Fn);
      const bool IsVirtualCall = MD && MD->isVirtual();
      // Don't use [[noreturn]], _Noreturn or [[no_builtin]] for a call to a
      // virtual function. These attributes are not inherited by overloads.
      if (!(AttrOnCallSite && IsVirtualCall)) {
        if (Fn->isNoReturn())
          FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoReturn);
        NBA = Fn->getAttr<NoBuiltinAttr>();
      }
      // Only place nomerge attribute on call sites, never functions. This
      // allows it to work on indirect virtual function calls.
      if (AttrOnCallSite && TargetDecl->hasAttr<NoMergeAttr>())
        FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoMerge);
    }

    // 'const', 'pure' and 'noalias' attributed functions are also nounwind.
    if (TargetDecl->hasAttr<ConstAttr>()) {
      auto ME = llvm::MemoryEffects::none();
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::Memory,
                                               ME.toIntValue());
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'const' functions have greater restrictions than
      // 'pure' functions, so they also cannot have infinite loops.
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::WillReturn);
    } else if (TargetDecl->hasAttr<PureAttr>()) {
      auto ME = llvm::MemoryEffects::readOnly();
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::Memory,
                                               ME.toIntValue());
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'pure' functions cannot have infinite loops.
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::WillReturn);
    } else if (TargetDecl->hasAttr<NoAliasAttr>()) {
      auto ME = llvm::MemoryEffects::argMemOnly();
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::Memory,
                                               ME.toIntValue());
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoUnwind);
    }
    if (TargetDecl->hasAttr<RestrictAttr>())
      RetAttrsBuilder.addAttribute(llvm::Attribute::NoAlias);
    if (TargetDecl->hasAttr<ReturnsNonNullAttr>() &&
        !CGM.getCodeGenOpts().NullPointerIsValid)
      RetAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
    if (TargetDecl->hasAttr<AnyX86NoCallerSavedRegistersAttr>())
      FuncAttrsBuilder.addPassThroughAttribute("no_caller_saved_registers",
                                               UnitAttr::get(Ctx));
    if (TargetDecl->hasAttr<AnyX86NoCfCheckAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoCfCheck);
    if (TargetDecl->hasAttr<LeafAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoCallback);

    HasOptnone = TargetDecl->hasAttr<OptimizeNoneAttr>();
    if (auto *AllocSize = TargetDecl->getAttr<AllocSizeAttr>()) {
      Optional<int> NumElemsParam;
      if (AllocSize->getNumElemsParam().isValid())
        NumElemsParam = AllocSize->getNumElemsParam().getLLVMIndex();
      uint64_t RawArgs = packAllocSizeArgs(
          AllocSize->getElemSizeParam().getLLVMIndex(), NumElemsParam);
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::AllocSize,
                                               RawArgs);
    }

    if (TargetDecl->hasAttr<OpenCLKernelAttr>()) {
      if (CGM.getLangOpts().OpenCLVersion <= 120) {
        // OpenCL v1.2 Work groups are always uniform
        FuncAttrsBuilder.addPassThroughAttribute("uniform-work-group-size",
                                                 StringAttr::get(Ctx, "true"));
      } else {
        // OpenCL v2.0 Work groups may be whether uniform or not.
        // '-cl-uniform-work-group-size' compile option gets a hint
        // to the compiler that the global work-size be a multiple of
        // the work-group size specified to clEnqueueNDRangeKernel
        // (i.e. work groups are uniform).
        FuncAttrsBuilder.addPassThroughAttribute(
            "uniform-work-group-size",
            StringAttr::get(
                Ctx, llvm::toStringRef(CGM.getCodeGenOpts().UniformWGSize)));
      }
    }
  }

  // Attach "no-builtins" attributes to:
  // * call sites: both `nobuiltin` and "no-builtins" or "no-builtin-<name>".
  // * definitions: "no-builtins" or "no-builtin-<name>" only.
  // The attributes can come from:
  // * LangOpts: -ffreestanding, -fno-builtin, -fno-builtin-<name>
  // * FunctionDecl attributes: __attribute__((no_builtin(...)))
  addNoBuiltinAttributes(FuncAttrsBuilder, CGM.getLangOpts(), NBA);

  // Collect function IR attributes based on global settiings.
  getDefaultFunctionAttributes(Name, HasOptnone, AttrOnCallSite,
                               FuncAttrsBuilder);

  // Override some default IR attributes based on declaration-specific
  // information.
  if (TargetDecl) {
    if (TargetDecl->hasAttr<NoSpeculativeLoadHardeningAttr>())
      FuncAttrsBuilder.removeAttribute(
          llvm::Attribute::SpeculativeLoadHardening);
    if (TargetDecl->hasAttr<SpeculativeLoadHardeningAttr>())
      FuncAttrsBuilder.addPassThroughAttribute(
          llvm::Attribute::SpeculativeLoadHardening);
    if (TargetDecl->hasAttr<NoSplitStackAttr>())
      FuncAttrsBuilder.removeAttribute("split-stack");
    if (TargetDecl->hasAttr<ZeroCallUsedRegsAttr>()) {
      // A function "__attribute__((...))" overrides the command-line flag.
      auto Kind =
          TargetDecl->getAttr<ZeroCallUsedRegsAttr>()->getZeroCallUsedRegs();
      FuncAttrsBuilder.removeAttribute("zero-call-used-regs");
      FuncAttrsBuilder.addPassThroughAttribute(
          "zero-call-used-regs",
          StringAttr::get(
              Ctx,
              ZeroCallUsedRegsAttr::ConvertZeroCallUsedRegsKindToStr(Kind)));
    }

    // Add NonLazyBind attribute to function declarations when -fno-plt is used.
    if (CGM.getCodeGenOpts().NoPLT) {
      if (auto *Fn = dyn_cast<FunctionDecl>(TargetDecl)) {
        if (!Fn->isDefined() && !AttrOnCallSite) {
          FuncAttrsBuilder.addPassThroughAttribute(
              llvm::Attribute::NonLazyBind);
        }
      }
    }
  }

  // Add "sample-profile-suffix-elision-policy" attribute for internal linkage
  // functions with -funique-internal-linkage-names.
  if (TargetDecl && CGM.getCodeGenOpts().UniqueInternalLinkageNames) {
    if (isa<FunctionDecl>(TargetDecl)) {
      if (CGM.getFunctionLinkage(CalleeInfo.getCalleeDecl()) ==
          llvm::GlobalValue::InternalLinkage)
        FuncAttrsBuilder.addPassThroughAttribute(
            "sample-profile-suffix-elision-policy",
            StringAttr::get(Ctx, "selected"));
    }
  }

  // Collect non-call-site function IR attributes from declaration-specific
  // information.
  if (!AttrOnCallSite) {
    if (TargetDecl && TargetDecl->hasAttr<CmseNSEntryAttr>())
      FuncAttrsBuilder.addPassThroughAttribute("cmse_nonsecure_entry",
                                               UnitAttr::get(Ctx));

    // Whether tail calls are enabled.
    auto ShouldDisableTailCalls = [&] {
      // Should this be honored in getDefaultFunctionAttributes?
      if (CGM.getCodeGenOpts().DisableTailCalls)
        return true;

      if (!TargetDecl)
        return false;

      if (TargetDecl->hasAttr<DisableTailCallsAttr>() ||
          TargetDecl->hasAttr<AnyX86InterruptAttr>())
        return true;

      if (CGM.getCodeGenOpts().NoEscapingBlockTailCalls) {
        if (const auto *BD = dyn_cast<BlockDecl>(TargetDecl))
          if (!BD->doesNotEscape())
            return true;
      }

      return false;
    };
    if (ShouldDisableTailCalls())
      FuncAttrsBuilder.addPassThroughAttribute("disable-tail-calls",
                                               StringAttr::get(Ctx, "true"));

    // CPU/feature overrides.  addDefaultFunctionDefinitionAttributes
    // handles these separately to set them based on the global defaults.
    getCPUAndFeaturesAttributes(CalleeInfo.getCalleeDecl(), FuncAttrsBuilder);
  }

  // Collect attributes from arguments and return values.
  ClangToLLVMArgMapping IRFunctionArgs(CGM.getContext(), FI);

  QualType RetTy = FI.getReturnType();
  const clang::CodeGen::ABIArgInfo &RetAI = FI.getReturnInfo();
  const llvm::DataLayout &DL = CGM.getDataLayout();

  // C++ explicitly makes returning undefined values UB. C's rule only applies
  // to used values, so we never mark them noundef for now.
  bool HasStrictReturn = CGM.getLangOpts().CPlusPlus;
  if (TargetDecl && HasStrictReturn) {
    if (const FunctionDecl *FDecl = dyn_cast<FunctionDecl>(TargetDecl))
      HasStrictReturn &= !FDecl->isExternC();
    else if (const VarDecl *VDecl = dyn_cast<VarDecl>(TargetDecl))
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
        RetAI.getKind() != clang::CodeGen::ABIArgInfo::Indirect &&
        determineNoUndef(RetTy, CGM.getTypes(), DL, RetAI))
      RetAttrsBuilder.addAttribute(llvm::Attribute::NoUndef);
  }

  switch (RetAI.getKind()) {
  case clang::CodeGen::ABIArgInfo::Extend:
    if (RetAI.isSignExt())
      RetAttrsBuilder.addAttribute(llvm::Attribute::SExt);
    else
      RetAttrsBuilder.addAttribute(llvm::Attribute::ZExt);
    LLVM_FALLTHROUGH;
  case clang::CodeGen::ABIArgInfo::Direct:
    if (RetAI.getInReg())
      RetAttrsBuilder.addAttribute(llvm::Attribute::InReg);
    break;
  case clang::CodeGen::ABIArgInfo::Ignore:
    break;

  case clang::CodeGen::ABIArgInfo::InAlloca:
  case clang::CodeGen::ABIArgInfo::Indirect: {
    AddPotentialArgAccess();
    break;
  }

  case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
    break;

  case clang::CodeGen::ABIArgInfo::Expand:
  case clang::CodeGen::ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");
  }

  if (!IsThunk) {
    // FIXME: fix this properly, https://reviews.llvm.org/D100388
    if (const auto *RefTy = RetTy->getAs<ReferenceType>()) {
      QualType PTy = RefTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        RetAttrsBuilder.addAttribute(
            llvm::Attribute::Dereferenceable,
            CGM.getMinimumObjectSize(PTy).getQuantity());
      if (CGM.getContext().getTargetAddressSpace(PTy.getAddressSpace()) == 0 &&
          !CGM.getCodeGenOpts().NullPointerIsValid)
        RetAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
      if (PTy->isObjectType()) {
        llvm::Align Alignment =
            CGM.getNaturalPointeeTypeAlignment(RetTy).getAsAlign();
        RetAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                     Alignment.value());
      }
    }
  }

  bool HasUsedSRet = false;
  SmallVector<mlir::NamedAttrList, 4> ArgAttrs(IRFunctionArgs.totalIRArgs());

  // Attach attributes to sret.
  if (IRFunctionArgs.hasSRetArg()) {
    mlirclang::AttrBuilder SRETAttrs(*Ctx);
    SRETAttrs.addAttribute(llvm::Attribute::StructRet, getMLIRType(RetTy));
    HasUsedSRet = true;
    if (RetAI.getInReg())
      SRETAttrs.addAttribute(llvm::Attribute::InReg);
    SRETAttrs.addAttribute(llvm::Attribute::Alignment,
                           RetAI.getIndirectAlign().getQuantity());
    ArgAttrs[IRFunctionArgs.getSRetArgNo()] = SRETAttrs.getAttributes();
  }

  // Attach attributes to inalloca argument.
  if (IRFunctionArgs.hasInallocaArg()) {
    mlirclang::AttrBuilder Attrs(*Ctx);
    assert(false && "TODO");
    //    Attrs.addAttribute(llvm::Attribute::InAlloca,
    //                     getMLIRType(FI.getArgStruct()));
    ArgAttrs[IRFunctionArgs.getInallocaArgNo()] = Attrs.getAttributes();
  }

  // Apply `nonnull`, `dereferencable(N)` and `align N` to the `this` argument,
  // unless this is a thunk function.
  // FIXME: fix this properly, https://reviews.llvm.org/D100388
  if (FI.isInstanceMethod() && !IRFunctionArgs.hasInallocaArg() &&
      !FI.arg_begin()->type->isVoidPointerType() && !IsThunk) {
    auto IRArgs = IRFunctionArgs.getIRArgs(0);

    assert(IRArgs.second == 1 && "Expected only a single `this` pointer.");

    mlirclang::AttrBuilder ParamAttrsBuilder(*Ctx);

    QualType ThisTy =
        FI.arg_begin()->type.castAs<clang::PointerType>()->getPointeeType();

    QualType QT = FI.arg_begin()->type;
    if (!CGM.getCodeGenOpts().NullPointerIsValid &&
        CGM.getContext().getTargetAddressSpace(QT.getAddressSpace()) == 0) {
      ParamAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
      ParamAttrsBuilder.addAttribute(
          llvm::Attribute::Dereferenceable,
          CGM.getMinimumObjectSize(ThisTy).getQuantity());
    } else {
      // FIXME dereferenceable should be correct here, regardless of
      // NullPointerIsValid. However, dereferenceable currently does not always
      // respect NullPointerIsValid and may imply nonnull and break the program.
      // See https://reviews.llvm.org/D66618 for discussions.
      ParamAttrsBuilder.addAttribute(
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
    ParamAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                   Alignment.value());
    ArgAttrs[IRArgs.first] = ParamAttrsBuilder.getAttributes();
  }

  unsigned ArgNo = 0;
  for (clang::CodeGen::CGFunctionInfo::const_arg_iterator I = FI.arg_begin(),
                                                          E = FI.arg_end();
       I != E; ++I, ++ArgNo) {
    QualType ParamType = I->type;
    const clang::CodeGen::ABIArgInfo &AI = I->info;
    mlirclang::AttrBuilder ParamAttrsBuilder(*Ctx);

    // Add attribute for padding argument, if necessary.
    if (IRFunctionArgs.hasPaddingArg(ArgNo)) {
      if (AI.getPaddingInReg()) {
        ArgAttrs[IRFunctionArgs.getPaddingArgNo(ArgNo)] =
            mlirclang::AttrBuilder(*Ctx)
                .addAttribute(llvm::Attribute::InReg)
                .getAttributes();
      }
    }

    // Decide whether the argument we're handling could be partially undef
    if (CGM.getCodeGenOpts().EnableNoundefAttrs &&
        determineNoUndef(ParamType, CGM.getTypes(), DL, AI)) {
      ParamAttrsBuilder.addAttribute(llvm::Attribute::NoUndef);
    }

    // 'restrict' -> 'noalias' is done in EmitFunctionProlog when we
    // have the corresponding parameter variable.  It doesn't make
    // sense to do it here because parameters are so messed up.
    switch (AI.getKind()) {
    case clang::CodeGen::ABIArgInfo::Extend:
      if (AI.isSignExt())
        ParamAttrsBuilder.addAttribute(llvm::Attribute::SExt);
      else
        ParamAttrsBuilder.addAttribute(llvm::Attribute::ZExt);
      LLVM_FALLTHROUGH;
    case clang::CodeGen::ABIArgInfo::Direct:
      if (ArgNo == 0 && FI.isChainCall())
        ParamAttrsBuilder.addAttribute(llvm::Attribute::Nest);
      else if (AI.getInReg())
        ParamAttrsBuilder.addAttribute(llvm::Attribute::InReg);
      ParamAttrsBuilder.addAttribute(llvm::Attribute::StackAlignment,
                                     AI.getDirectAlign());
      break;

    case clang::CodeGen::ABIArgInfo::Indirect: {
      if (AI.getInReg())
        ParamAttrsBuilder.addAttribute(llvm::Attribute::InReg);

      if (AI.getIndirectByVal())
        ParamAttrsBuilder.addAttribute(llvm::Attribute::ByVal,
                                       getMLIRType(ParamType));

      auto *Decl = ParamType->getAsRecordDecl();
      if (CGM.getCodeGenOpts().PassByValueIsNoAlias && Decl &&
          Decl->getArgPassingRestrictions() == RecordDecl::APK_CanPassInRegs)
        // When calling the function, the pointer passed in will be the only
        // reference to the underlying object. Mark it accordingly.
        ParamAttrsBuilder.addAttribute(llvm::Attribute::NoAlias);

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
        ParamAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                       Align.getQuantity());

      // byval disables readnone and readonly.
      AddPotentialArgAccess();

      break;
    }
    case clang::CodeGen::ABIArgInfo::IndirectAliased: {
      CharUnits Align = AI.getIndirectAlign();
      ParamAttrsBuilder
          .addAttribute(llvm::Attribute::ByRef, getMLIRType(ParamType))
          .addAttribute(llvm::Attribute::Alignment, Align.getQuantity());
      break;
    }
    case clang::CodeGen::ABIArgInfo::Ignore:
    case clang::CodeGen::ABIArgInfo::Expand:
    case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
      break;

    case clang::CodeGen::ABIArgInfo::InAlloca:
      AddPotentialArgAccess();
      continue;
    }

    if (const auto *RefTy = ParamType->getAs<ReferenceType>()) {
      QualType PTy = RefTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        ParamAttrsBuilder.addAttribute(
            llvm::Attribute::Dereferenceable,
            CGM.getMinimumObjectSize(PTy).getQuantity());
      if (CGM.getContext().getTargetAddressSpace(PTy.getAddressSpace()) == 0 &&
          !CGM.getCodeGenOpts().NullPointerIsValid)
        ParamAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
      if (PTy->isObjectType()) {
        llvm::Align Alignment =
            CGM.getNaturalPointeeTypeAlignment(ParamType).getAsAlign();
        ParamAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                       Alignment.value());
      }
    }

    // From OpenCL spec v3.0.10 section 6.3.5 Alignment of Types:
    // > For arguments to a __kernel function declared to be a pointer to a
    // > data type, the OpenCL compiler can assume that the pointee is always
    // > appropriately aligned as required by the data type.
    if (TargetDecl && TargetDecl->hasAttr<OpenCLKernelAttr>() &&
        ParamType->isPointerType()) {
      QualType PTy = ParamType->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType()) {
        CharUnits Alignment = CGM.getNaturalPointeeTypeAlignment(ParamType);
        ParamAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                       Alignment.getQuantity());
      }
    }

    switch (FI.getExtParameterInfo(ArgNo).getABI()) {
    case ParameterABI::Ordinary:
      break;

    case ParameterABI::SwiftIndirectResult: {
      // Add 'sret' if we haven't already used it for something, but
      // only if the result is void.
      if (!HasUsedSRet && RetTy->isVoidType()) {
        ParamAttrsBuilder.addAttribute(llvm::Attribute::StructRet,
                                       getMLIRType(ParamType));
        HasUsedSRet = true;
      }

      // Add 'noalias' in either case.
      ParamAttrsBuilder.addAttribute(llvm::Attribute::NoAlias);

      // Add 'dereferenceable' and 'alignment'.
      auto PTy = ParamType->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType()) {
        auto Info = CGM.getContext().getTypeInfoInChars(PTy);
        ParamAttrsBuilder.addAttribute(llvm::Attribute::Dereferenceable,
                                       Info.Width.getQuantity());
        ParamAttrsBuilder.addAttribute(llvm::Attribute::Alignment,
                                       Info.Align.getQuantity());
      }
      break;
    }

    case ParameterABI::SwiftErrorResult:
      ParamAttrsBuilder.addAttribute(llvm::Attribute::SwiftError);
      break;

    case ParameterABI::SwiftContext:
      ParamAttrsBuilder.addAttribute(llvm::Attribute::SwiftSelf);
      break;

    case ParameterABI::SwiftAsyncContext:
      ParamAttrsBuilder.addAttribute(llvm::Attribute::SwiftAsync);
      break;
    }

    if (FI.getExtParameterInfo(ArgNo).isNoEscape())
      ParamAttrsBuilder.addAttribute(llvm::Attribute::NoCapture);

    if (ParamAttrsBuilder.hasAttributes()) {
      unsigned FirstIRArg, NumIRArgs;
      std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);
      for (unsigned I = 0; I < NumIRArgs; I++)
        ArgAttrs[FirstIRArg + I].append(ParamAttrsBuilder.getAttributes());
    }
  }
  assert(ArgNo == FI.arg_size());

  AttrList.addAttrs(FuncAttrsBuilder, RetAttrsBuilder, ArgAttrs);
}

mlir::Type CodeGenTypes::getMLIRType(clang::QualType QT, bool *ImplicitRef,
                                     bool AllowMerge) {
  if (const auto *ET = dyn_cast<clang::ElaboratedType>(QT))
    return getMLIRType(ET->getNamedType(), ImplicitRef, AllowMerge);

  if (const auto *ET = dyn_cast<clang::UsingType>(QT))
    return getMLIRType(ET->getUnderlyingType(), ImplicitRef, AllowMerge);

  if (const auto *ET = dyn_cast<clang::ParenType>(QT))
    return getMLIRType(ET->getInnerType(), ImplicitRef, AllowMerge);

  if (const auto *ET = dyn_cast<clang::DeducedType>(QT))
    return getMLIRType(ET->getDeducedType(), ImplicitRef, AllowMerge);

  if (const auto *ST = dyn_cast<clang::SubstTemplateTypeParmType>(QT))
    return getMLIRType(ST->getReplacementType(), ImplicitRef, AllowMerge);

  if (const auto *ST = dyn_cast<clang::TemplateSpecializationType>(QT))
    return getMLIRType(ST->desugar(), ImplicitRef, AllowMerge);

  if (const auto *ST = dyn_cast<clang::TypedefType>(QT))
    return getMLIRType(ST->desugar(), ImplicitRef, AllowMerge);

  if (const auto *DT = dyn_cast<clang::DecltypeType>(QT))
    return getMLIRType(DT->desugar(), ImplicitRef, AllowMerge);

  if (const auto *DT = dyn_cast<clang::DecayedType>(QT)) {
    bool AssumeRef = false;
    auto MLIRTy = getMLIRType(DT->getOriginalType(), &AssumeRef, AllowMerge);
    if (MemRefABI && AssumeRef) {
      // Constant array types like `int A[30][20]` will be converted to LLVM
      // type `[20 x i32]* %0`, which has the outermost dimension size erased,
      // and we can only recover to `memref<?x20xi32>` from there. This
      // prevents us from doing more comprehensive analysis.Here we
      // specifically handle this case by unwrapping the clang-adjusted
      // type, to get the corresponding ConstantArrayType with the full
      // dimensions.
      if (MemRefFullRank) {
        clang::QualType OrigTy = DT->getOriginalType();
        if (OrigTy->isConstantArrayType()) {
          SmallVector<int64_t, 4> Shape;
          clang::QualType ElemTy;
          getConstantArrayShapeAndElemType(OrigTy, Shape, ElemTy);
          return mlir::MemRefType::get(Shape, getMLIRType(ElemTy));
        }
      }

      // If -memref-fullrank is unset or it cannot be fulfilled.
      auto MT = MLIRTy.dyn_cast<MemRefType>();
      auto Shape2 = std::vector<int64_t>(MT.getShape());
      Shape2[0] = ShapedType::kDynamic;
      return mlir::MemRefType::get(Shape2, MT.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   MT.getMemorySpace());
    }
    return getMLIRType(DT->getAdjustedType(), ImplicitRef, AllowMerge);
  }

  if (const auto *CT = dyn_cast<clang::ComplexType>(QT)) {
    bool AssumeRef = false;
    auto SubType =
        getMLIRType(CT->getElementType(), &AssumeRef, /*AllowMerge*/ false);
    if (MemRefABI && AllowMerge) {
      assert(!AssumeRef);
      if (ImplicitRef)
        *ImplicitRef = true;
      return mlir::MemRefType::get(2, SubType);
    }
    mlir::Type Types[2] = {SubType, SubType};
    return mlir::LLVM::LLVMStructType::getLiteral(TheModule->getContext(),
                                                  Types);
  }

  mlir::LLVM::TypeFromLLVMIRTranslator TypeTranslator(*TheModule->getContext());

  if (const auto *RT = dyn_cast<clang::RecordType>(QT)) {
    LLVM_DEBUG({
      if (RT->getDecl()->isInvalidDecl()) {
        RT->getDecl()->dump();
        RT->dump();
      }
    });
    assert(!RT->getDecl()->isInvalidDecl());

    if (TypeCache.find(RT) != TypeCache.end())
      return TypeCache[RT];

    llvm::Type *LT = CGM.getTypes().ConvertType(QT);
    LLVM_DEBUG({
      if (!isa<llvm::StructType>(LT)) {
        QT->dump();
        llvm::errs() << "LT: " << *LT << "\n";
      }
    });
    llvm::StructType *ST = cast<llvm::StructType>(LT);

    bool Recursive = false;
    for (size_t I = 0; I < ST->getNumElements(); I++) {
      SmallPtrSet<llvm::Type *, 4> Seen;
      if (isRecursiveStruct(ST->getTypeAtIndex(I), ST, Seen))
        Recursive = true;
    }

    const auto *RD = RT->getAsRecordDecl();
    if (const mlirclang::NamespaceKind NamespaceKind =
            mlirclang::getNamespaceKind(RD->getEnclosingNamespaceContext());
        NamespaceKind != mlirclang::NamespaceKind::Other) {
      const auto TypeName = RD->getName();
      if (TypeName == "accessor" || TypeName == "accessor_common" ||
          TypeName == "AccessorImplDevice" || TypeName == "AccessorSubscript" ||
          TypeName == "array" || TypeName == "AssertHappened" ||
          TypeName == "atomic" || TypeName == "bfloat16" ||
          TypeName == "GetOp" || TypeName == "GetScalarOp" ||
          TypeName == "group" || TypeName == "h_item" || TypeName == "id" ||
          TypeName == "item" || TypeName == "ItemBase" ||
          TypeName == "kernel_handler" ||
          TypeName == "LocalAccessorBaseDevice" ||
          TypeName == "local_accessor_base" || TypeName == "local_accessor" ||
          TypeName == "maximum" || TypeName == "minimum" ||
          TypeName == "multi_ptr" || TypeName == "nd_item" ||
          TypeName == "nd_range" || TypeName == "OwnerLessBase" ||
          TypeName == "range" || TypeName == "stream" ||
          TypeName == "sub_group" || TypeName == "SwizzleOp" ||
          TypeName == "TupleCopyAssignableValueHolder" ||
          TypeName == "TupleValueHolder" || TypeName == "vec")
        return getSYCLType(RT, *this);

      assert((AllowUndefinedSYCLTypes ||
              NamespaceKind != mlirclang::NamespaceKind::SYCL) &&
             "Found type in the sycl namespace, but not in the SYCL dialect");
    }

    auto *CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (CodeGenTypes::isLLVMStructABI(RT->getDecl(), ST))
      return TypeTranslator.translateType(anonymize(ST));

    if (CXRD && CXRD->isUnion()) {
      assert(isSingleFieldUnion(CXRD) &&
             "Only handling single-field enumerations");
      return LLVM::LLVMStructType::getLiteral(
          TheModule->getContext(), getMLIRType(CXRD->field_begin()->getType()));
    }

    if (Recursive)
      TypeCache[RT] = LLVM::LLVMStructType::getIdentified(
          TheModule->getContext(), ("polygeist@mlir@" + ST->getName()).str());

    SmallVector<mlir::Type, 4> Types;

    bool InnerLLVM = false;
    bool InnerSYCL = false;
    if (CXRD) {
      for (auto F : CXRD->bases()) {
        bool SubRef = false;
        auto Ty = getMLIRType(F.getType(), &SubRef, /*AllowMerge*/ false);
        assert(!SubRef);
        InnerLLVM |= Ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                            LLVM::LLVMArrayType>();
        Types.push_back(Ty);
      }
    }

    for (auto *F : RT->getDecl()->fields()) {
      bool SubRef = false;
      auto Ty = getMLIRType(F->getType(), &SubRef, /*AllowMerge*/ false);
      assert(!SubRef);
      InnerLLVM |= Ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                          LLVM::LLVMArrayType>();
      InnerSYCL |= mlir::sycl::isSYCLType(Ty);
      Types.push_back(Ty);
    }

    if (Types.empty())
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return TypeTranslator.translateType(anonymize(ST));

    if (Recursive) {
      auto LR = TypeCache[RT].setBody(Types, /*isPacked*/ false);
      assert(LR.succeeded());
      return TypeCache[RT];
    }

    return LLVM::LLVMStructType::getLiteral(TheModule->getContext(), Types);
  }

  const clang::Type *T = QT->getUnqualifiedDesugaredType();

  if (const auto *AT = dyn_cast<clang::ArrayType>(T)) {
    const auto *PTT = AT->getElementType()->getUnqualifiedDesugaredType();
    if (PTT->isCharType()) {
      llvm::Type *Ty = CGM.getTypes().ConvertType(QualType(T, 0));
      return TypeTranslator.translateType(Ty);
    }

    bool SubRef = false;
    auto ET = getMLIRType(AT->getElementType(), &SubRef, AllowMerge);
    int64_t Size = ShapedType::kDynamic;
    if (const auto *CAT = dyn_cast<clang::ConstantArrayType>(AT))
      Size = CAT->getSize().getZExtValue();
    if (MemRefABI && SubRef) {
      auto MT = ET.cast<MemRefType>();
      auto Shape2 = std::vector<int64_t>(MT.getShape());
      Shape2.insert(Shape2.begin(), Size);
      if (ImplicitRef)
        *ImplicitRef = true;
      return mlir::MemRefType::get(Shape2, MT.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   MT.getMemorySpace());
    }

    if (!MemRefABI || !AllowMerge ||
        ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
               LLVM::LLVMFunctionType, LLVM::LLVMStructType>())
      return LLVM::LLVMArrayType::get(
          ET, (Size == ShapedType::kDynamic) ? 0 : Size);

    if (ImplicitRef)
      *ImplicitRef = true;

    return mlir::MemRefType::get({Size}, ET, {},
                                 CGM.getContext().getTargetAddressSpace(
                                     AT->getElementType().getAddressSpace()));
  }

  if (const auto *AT = dyn_cast<clang::VectorType>(T)) {
    bool SubRef = false;
    auto ET = getMLIRType(AT->getElementType(), &SubRef, AllowMerge);
    int64_t Size = AT->getNumElements();
    return mlir::VectorType::get(Size, ET);
  }

  if (const auto *FT = dyn_cast<clang::FunctionProtoType>(T)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (RT.isa<mlir::NoneType>())
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    for (auto T : FT->getParamTypes())
      Args.push_back(getMLIRType(T));
    return LLVM::LLVMFunctionType::get(RT, Args, FT->isVariadic());
  }

  if (const auto *FT = dyn_cast<clang::FunctionNoProtoType>(T)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (RT.isa<mlir::NoneType>())
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    return LLVM::LLVMFunctionType::get(RT, Args, /*isVariadic*/ true);
  }

  if (isa<clang::PointerType, clang::ReferenceType>(T)) {
    int64_t Outer = ShapedType::kDynamic;
    auto PointeeType = isa<clang::PointerType>(T)
                           ? cast<clang::PointerType>(T)->getPointeeType()
                           : cast<clang::ReferenceType>(T)->getPointeeType();
    const clang::Type *PTT = PointeeType->getUnqualifiedDesugaredType();

    if (PTT->isVoidType()) {
      llvm::Type *Ty = CGM.getTypes().ConvertType(QualType(T, 0));
      return TypeTranslator.translateType(Ty);
    }

    bool SubRef = false;
    auto SubType = getMLIRType(PointeeType, &SubRef, /*AllowMerge*/ true);

    if (!MemRefABI ||
        SubType.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType,
                    LLVM::LLVMPointerType, LLVM::LLVMFunctionType>()) {
      // JLE_QUEL::THOUGHTS
      // When generating the sycl_halide_kernel, If a struct type contains
      // SYCL types, that means that this is the functor, and we can't create
      // a llvm pointer that contains custom aggregate types. We could create
      // a sycl::Functor type, that will help us get rid of those conditions.
      bool InnerSYCL = false;
      if (auto ST = SubType.dyn_cast<mlir::LLVM::LLVMStructType>())
        InnerSYCL |= any_of(ST.getBody(), mlir::sycl::isSYCLType);

      if (!InnerSYCL)
        return LLVM::LLVMPointerType::get(
            SubType, CGM.getContext().getTargetAddressSpace(
                         PointeeType.getAddressSpace()));
    }

    if (isa<clang::ArrayType>(PTT)) {
      if (SubType.isa<MemRefType>()) {
        assert(SubRef);
        return SubType;
      }
      return LLVM::LLVMPointerType::get(SubType);
    }

    if (isa<clang::VectorType>(PTT) || isa<clang::ComplexType>(PTT)) {
      if (auto VT = SubType.dyn_cast<mlir::VectorType>())
        // FIXME: We should create memref of rank 0.
        // Details: https://github.com/intel/llvm/issues/7354
        return mlir::MemRefType::get(Outer, SubType);

      if (SubType.isa<MemRefType>()) {
        assert(SubRef);
        auto MT = SubType.cast<MemRefType>();
        auto Shape2 = std::vector<int64_t>(MT.getShape());
        Shape2.insert(Shape2.begin(), Outer);
        return mlir::MemRefType::get(Shape2, MT.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     MT.getMemorySpace());
      }

      return LLVM::LLVMPointerType::get(SubType);
    }

    if (isa<clang::RecordType>(PTT) && SubRef) {
      auto MT = SubType.cast<MemRefType>();
      auto Shape2 = std::vector<int64_t>(MT.getShape());
      Shape2.insert(Shape2.begin(), Outer);
      return mlir::MemRefType::get(Shape2, MT.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   MT.getMemorySpace());
    }
    assert(!SubRef);

    return mlir::MemRefType::get(
        {Outer}, SubType, {},
        CGM.getContext().getTargetAddressSpace(PointeeType.getAddressSpace()));
  }

  if (isa<clang::EnumType>(T)) {
    mlir::OpBuilder Builder(TheModule->getContext());
    llvm::Type *Ty = CGM.getTypes().ConvertType(QualType(T, 0));
    return Builder.getIntegerType(cast<llvm::IntegerType>(Ty)->getBitWidth());
  }

  if (T->isBuiltinType())
    return getMLIRType(cast<clang::BuiltinType>(T));

  LLVM_DEBUG(llvm::dbgs() << "QT: "; QT->dump(); llvm::dbgs() << "\n");
  llvm_unreachable("unhandled type");
}

mlir::Type CodeGenTypes::getMLIRType(const clang::BuiltinType *BT) const {
  assert(BT && "Expecting valid pointer");

  mlir::OpBuilder Builder(TheModule->getContext());
  mlir::LLVM::TypeFromLLVMIRTranslator TypeTranslator(*TheModule->getContext());

  switch (BT->getKind()) {
  case BuiltinType::Void:
    return Builder.getNoneType();

  case BuiltinType::ObjCId:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCSel:
    return Builder.getIntegerType(8);

  case BuiltinType::Bool:
    // TODO: boolean types should be represented as i1 rather than i8.
    return Builder.getIntegerType(8);

  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
    return Builder.getIntegerType(Context.getTypeSize(BT));

  case BuiltinType::Float16:
  case BuiltinType::Half:
  case BuiltinType::BFloat16:
    return Builder.getF16Type();

  case BuiltinType::Float:
    return Builder.getF32Type();

  case BuiltinType::Double:
    return Builder.getF64Type();

  case BuiltinType::LongDouble:
  case BuiltinType::Float128:
  case BuiltinType::Ibm128:
    return Builder.getF128Type();

  case BuiltinType::NullPtr:
    // Model std::nullptr_t as i8*
    return getPointerOrMemRefType(Builder.getIntegerType(8), 0);

  case BuiltinType::UInt128:
  case BuiltinType::Int128:
    return Builder.getIntegerType(128);

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Sampled##Id:
#define IMAGE_WRITE_TYPE(Type, Id, Ext)
#define IMAGE_READ_WRITE_TYPE(Type, Id, Ext)
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLClkEvent:
  case BuiltinType::OCLQueue:
  case BuiltinType::OCLReserveID:
    return TypeTranslator.translateType(
        CGM.getOpenCLRuntime().convertOpenCLSpecificType(BT));

  case BuiltinType::SveInt8:
  case BuiltinType::SveUint8:
  case BuiltinType::SveInt8x2:
  case BuiltinType::SveUint8x2:
  case BuiltinType::SveInt8x3:
  case BuiltinType::SveUint8x3:
  case BuiltinType::SveInt8x4:
  case BuiltinType::SveUint8x4:
  case BuiltinType::SveInt16:
  case BuiltinType::SveUint16:
  case BuiltinType::SveInt16x2:
  case BuiltinType::SveUint16x2:
  case BuiltinType::SveInt16x3:
  case BuiltinType::SveUint16x3:
  case BuiltinType::SveInt16x4:
  case BuiltinType::SveUint16x4:
  case BuiltinType::SveInt32:
  case BuiltinType::SveUint32:
  case BuiltinType::SveInt32x2:
  case BuiltinType::SveUint32x2:
  case BuiltinType::SveInt32x3:
  case BuiltinType::SveUint32x3:
  case BuiltinType::SveInt32x4:
  case BuiltinType::SveUint32x4:
  case BuiltinType::SveInt64:
  case BuiltinType::SveUint64:
  case BuiltinType::SveInt64x2:
  case BuiltinType::SveUint64x2:
  case BuiltinType::SveInt64x3:
  case BuiltinType::SveUint64x3:
  case BuiltinType::SveInt64x4:
  case BuiltinType::SveUint64x4:
  case BuiltinType::SveBool:
  case BuiltinType::SveFloat16:
  case BuiltinType::SveFloat16x2:
  case BuiltinType::SveFloat16x3:
  case BuiltinType::SveFloat16x4:
  case BuiltinType::SveFloat32:
  case BuiltinType::SveFloat32x2:
  case BuiltinType::SveFloat32x3:
  case BuiltinType::SveFloat32x4:
  case BuiltinType::SveFloat64:
  case BuiltinType::SveFloat64x2:
  case BuiltinType::SveFloat64x3:
  case BuiltinType::SveFloat64x4:
  case BuiltinType::SveBFloat16:
  case BuiltinType::SveBFloat16x2:
  case BuiltinType::SveBFloat16x3:
  case BuiltinType::SveBFloat16x4:
    llvm_unreachable("Unexpected ARM type");

#define PPC_VECTOR_TYPE(Name, Id, Size) case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
    llvm_unreachable("Unexpected PPC type");

  case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
    llvm_unreachable("Unexpected placeholder builtin type!");
  }

  llvm_unreachable("Unexpected builtin type!");
}

// Note: In principle we should always create a memref here because we want to
// avoid lowering the abstraction level at this point in the compilation flow.
// However, cgeist treats type inconsistently, it expects memref for SYCL
// types and pointers for every other struct type.
mlir::Type CodeGenTypes::getPointerOrMemRefType(mlir::Type Ty,
                                                unsigned AddressSpace,
                                                bool IsAlloc) const {
  auto ST = Ty.dyn_cast<mlir::LLVM::LLVMStructType>();

  bool IsSYCLType = mlir::sycl::isSYCLType(Ty);
  if (ST)
    IsSYCLType |= any_of(ST.getBody(), mlir::sycl::isSYCLType);

  if (!ST || IsSYCLType)
    return mlir::MemRefType::get(IsAlloc ? 1 : ShapedType::kDynamic, Ty, {},
                                 AddressSpace);

  return LLVM::LLVMPointerType::get(Ty, AddressSpace);
}

const clang::CodeGen::CGFunctionInfo &
CodeGenTypes::arrangeGlobalDeclaration(clang::GlobalDecl GD) {
  return CGM.getTypes().arrangeGlobalDeclaration(GD);
}

bool CodeGenTypes::isLLVMStructABI(const clang::RecordDecl *RD,
                                   llvm::StructType *ST) {
  if (!CombinedStructABI)
    return true;

  if (RD->isUnion()) {
    // We will handle single-field unions as non-LLVMStructABI types.
    return !isSingleFieldUnion(RD);
  }

  if (const auto *CXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (!CXRD->hasDefinition() || CXRD->getNumVBases())
      return true;
    for (const auto *M : CXRD->methods())
      if (M->isVirtualAsWritten() || M->isPure())
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

void CodeGenTypes::getDefaultFunctionAttributes(
    StringRef Name, bool HasOptnone, bool AttrOnCallSite,
    mlirclang::AttrBuilder &FuncAttrs) const {
  MLIRContext *Ctx = TheModule->getContext();
  const LangOptions &LangOpts = CGM.getLangOpts();

  // OptimizeNoneAttr takes precedence over -Os or -Oz. No warning needed.
  if (!HasOptnone) {
    if (getCodeGenOpts().OptimizeSize)
      FuncAttrs.addPassThroughAttribute(llvm::Attribute::OptimizeForSize);
    if (getCodeGenOpts().OptimizeSize == 2)
      FuncAttrs.addPassThroughAttribute(llvm::Attribute::MinSize);
  }

  if (getCodeGenOpts().DisableRedZone)
    FuncAttrs.addPassThroughAttribute(llvm::Attribute::NoRedZone);
  if (getCodeGenOpts().IndirectTlsSegRefs)
    FuncAttrs.addPassThroughAttribute("indirect-tls-seg-refs",
                                      UnitAttr::get(Ctx));
  if (getCodeGenOpts().NoImplicitFloat)
    FuncAttrs.addPassThroughAttribute(llvm::Attribute::NoImplicitFloat);

  if (AttrOnCallSite) {
    // Attributes that should go on the call site only.
    // FIXME: Look for 'BuiltinAttr' on the function rather than re-checking
    // the -fno-builtin-foo list.
    if (!getCodeGenOpts().SimplifyLibCalls || LangOpts.isNoBuiltinFunc(Name))
      FuncAttrs.addPassThroughAttribute(llvm::Attribute::NoBuiltin);
    if (!getCodeGenOpts().TrapFuncName.empty())
      FuncAttrs.addPassThroughAttribute(
          "trap-func-name",
          StringAttr::get(Ctx, getCodeGenOpts().TrapFuncName));
  } else {
    StringRef FpKind;
    switch (getCodeGenOpts().getFramePointer()) {
    case CodeGenOptions::FramePointerKind::None:
      FpKind = "none";
      break;
    case CodeGenOptions::FramePointerKind::NonLeaf:
      FpKind = "non-leaf";
      break;
    case CodeGenOptions::FramePointerKind::All:
      FpKind = "all";
      break;
    }
    FuncAttrs.addPassThroughAttribute("frame-pointer",
                                      StringAttr::get(Ctx, FpKind));

    if (getCodeGenOpts().LessPreciseFPMAD)
      FuncAttrs.addPassThroughAttribute("less-precise-fpmad",
                                        StringAttr::get(Ctx, "true"));

    if (getCodeGenOpts().NullPointerIsValid)
      FuncAttrs.addPassThroughAttribute(llvm::Attribute::NullPointerIsValid);

    if (getCodeGenOpts().FPDenormalMode != llvm::DenormalMode::getIEEE())
      FuncAttrs.addPassThroughAttribute(
          "denormal-fp-math",
          StringAttr::get(Ctx, getCodeGenOpts().FPDenormalMode.str()));
    if (getCodeGenOpts().FP32DenormalMode != getCodeGenOpts().FPDenormalMode) {
      FuncAttrs.addPassThroughAttribute(
          "denormal-fp-math-f32",
          StringAttr::get(Ctx, getCodeGenOpts().FP32DenormalMode.str()));
    }

    if (LangOpts.getDefaultExceptionMode() == LangOptions::FPE_Ignore)
      FuncAttrs.addPassThroughAttribute("no-trapping-math",
                                        StringAttr::get(Ctx, "true"));

    // TODO: Are these all needed?
    // unsafe/inf/nan/nsz are handled by instruction-level FastMathFlags.
    if (LangOpts.NoHonorInfs)
      FuncAttrs.addPassThroughAttribute("no-infs-fp-math",
                                        StringAttr::get(Ctx, "true"));
    if (LangOpts.NoHonorNaNs)
      FuncAttrs.addPassThroughAttribute("no-nans-fp-math",
                                        StringAttr::get(Ctx, "true"));
    if (LangOpts.ApproxFunc)
      FuncAttrs.addPassThroughAttribute("approx-func-fp-math",
                                        StringAttr::get(Ctx, "true"));
    if (LangOpts.UnsafeFPMath)
      FuncAttrs.addPassThroughAttribute("unsafe-fp-math",
                                        StringAttr::get(Ctx, "true"));
    if (getCodeGenOpts().SoftFloat)
      FuncAttrs.addPassThroughAttribute("use-soft-float",
                                        StringAttr::get(Ctx, "true"));
    FuncAttrs.addPassThroughAttribute(
        "stack-protector-buffer-size",
        StringAttr::get(Ctx, llvm::utostr(getCodeGenOpts().SSPBufferSize)));
    if (LangOpts.NoSignedZero)
      FuncAttrs.addPassThroughAttribute("no-signed-zeros-fp-math",
                                        StringAttr::get(Ctx, "true"));

    // TODO: Reciprocal estimate codegen options should apply to instructions?
    const std::vector<std::string> &Recips = getCodeGenOpts().Reciprocals;
    if (!Recips.empty())
      FuncAttrs.addPassThroughAttribute(
          "reciprocal-estimates",
          StringAttr::get(Ctx, llvm::join(Recips, ",")));

    if (!getCodeGenOpts().PreferVectorWidth.empty() &&
        getCodeGenOpts().PreferVectorWidth != "none")
      FuncAttrs.addPassThroughAttribute(
          "prefer-vector-width",
          StringAttr::get(Ctx, getCodeGenOpts().PreferVectorWidth));

    if (getCodeGenOpts().StackRealignment)
      FuncAttrs.addPassThroughAttribute("stackrealign", UnitAttr::get(Ctx));
    if (getCodeGenOpts().Backchain)
      FuncAttrs.addPassThroughAttribute("backchain", UnitAttr::get(Ctx));
    if (getCodeGenOpts().EnableSegmentedStacks)
      FuncAttrs.addPassThroughAttribute("split-stack", UnitAttr::get(Ctx));

    if (getCodeGenOpts().SpeculativeLoadHardening)
      FuncAttrs.addPassThroughAttribute(
          llvm::Attribute::SpeculativeLoadHardening);

    // Add zero-call-used-regs attribute.
    switch (getCodeGenOpts().getZeroCallUsedRegs()) {
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Skip:
      FuncAttrs.removeAttribute("zero-call-used-regs");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedGPRArg:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "used-gpr-arg"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedGPR:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "used-gpr"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedArg:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "used-arg"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Used:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "used"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllGPRArg:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "all-gpr-arg"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllGPR:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "all-gpr"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllArg:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "all-arg"));
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::All:
      FuncAttrs.addPassThroughAttribute("zero-call-used-regs",
                                        StringAttr::get(Ctx, "all"));
      break;
    }
  }

  if (LangOpts.assumeFunctionsAreConvergent()) {
    // Conservatively, mark all functions and calls in CUDA and OpenCL as
    // convergent (meaning, they may call an intrinsically convergent op, such
    // as __syncthreads() / barrier(), and so can't have certain optimizations
    // applied around them).  LLVM will remove this attribute where it safely
    // can.
    FuncAttrs.addPassThroughAttribute(llvm::Attribute::Convergent);
  }

  // TODO: NoUnwind attribute should be added for other GPU modes OpenCL, HIP,
  // OpenMP offload. AFAIK, none of them support exceptions in device code.
  if ((LangOpts.CUDA && LangOpts.CUDAIsDevice) ||
      (LangOpts.isSYCL() && LangOpts.SYCLIsDevice))
    FuncAttrs.addPassThroughAttribute(llvm::Attribute::NoUnwind);

  for (StringRef Attr : CGM.getCodeGenOpts().DefaultFunctionAttrs) {
    StringRef Var, Value;
    std::tie(Var, Value) = Attr.split('=');
    FuncAttrs.addPassThroughAttribute(Var, StringAttr::get(Ctx, Value));
  }
}

bool CodeGenTypes::getCPUAndFeaturesAttributes(
    GlobalDecl GD, AttrBuilder &FuncAttrsBuilder) const {
  // Add target-cpu and target-features attributes to functions. If
  // we have a decl for the function and it has a target attribute then
  // parse that and add it to the feature set.
  StringRef TargetCPU = CGM.getTarget().getTargetOpts().CPU;
  StringRef TuneCPU = CGM.getTarget().getTargetOpts().TuneCPU;
  std::vector<std::string> Features;
  const auto *FD = dyn_cast_or_null<FunctionDecl>(GD.getDecl());
  FD = FD ? FD->getMostRecentDecl() : FD;
  const auto *TD = FD ? FD->getAttr<TargetAttr>() : nullptr;
  const auto *SD = FD ? FD->getAttr<CPUSpecificAttr>() : nullptr;
  const auto *TC = FD ? FD->getAttr<TargetClonesAttr>() : nullptr;
  bool AddedAttr = false;
  if (TD || SD || TC) {
    llvm::StringMap<bool> FeatureMap;
    getContext().getFunctionFeatureMap(FeatureMap, GD);

    // Produce the canonical string for this set of features.
    for (const llvm::StringMap<bool>::value_type &Entry : FeatureMap)
      Features.push_back((Entry.getValue() ? "+" : "-") + Entry.getKey().str());

    // Now add the target-cpu and target-features to the function.
    // While we populated the feature map above, we still need to
    // get and parse the target attribute so we can get the cpu for
    // the function.
    if (TD) {
      ParsedTargetAttr ParsedAttr =
          CGM.getTarget().parseTargetAttr(TD->getFeaturesStr());
      if (!ParsedAttr.CPU.empty() &&
          CGM.getTarget().isValidCPUName(ParsedAttr.CPU)) {
        TargetCPU = ParsedAttr.CPU;
        TuneCPU = ""; // Clear the tune CPU.
      }
      if (!ParsedAttr.Tune.empty() &&
          CGM.getTarget().isValidCPUName(ParsedAttr.Tune))
        TuneCPU = ParsedAttr.Tune;
    }

    if (SD) {
      // Apply the given CPU name as the 'tune-cpu' so that the optimizer can
      // favor this processor.
      TuneCPU = CGM.getTarget().getCPUSpecificTuneName(
          SD->getCPUName(GD.getMultiVersionIndex())->getName());
    }
  } else {
    // Otherwise just add the existing target cpu and target features to the
    // function.
    Features = CGM.getTarget().getTargetOpts().Features;
  }

  MLIRContext *Ctx = TheModule->getContext();

  if (!TargetCPU.empty()) {
    FuncAttrsBuilder.addPassThroughAttribute("target-cpu",
                                             StringAttr::get(Ctx, TargetCPU));
    AddedAttr = true;
  }
  if (!TuneCPU.empty()) {
    FuncAttrsBuilder.addPassThroughAttribute("tune-cpu",
                                             StringAttr::get(Ctx, TuneCPU));
    AddedAttr = true;
  }
  if (!Features.empty()) {
    llvm::sort(Features);
    FuncAttrsBuilder.addPassThroughAttribute(
        "target-features", StringAttr::get(Ctx, llvm::join(Features, ",")));
    AddedAttr = true;
  }

  return AddedAttr;
}

QualType CodeGenTypes::getPromotionType(QualType Ty) const {
  if (CGM.getTarget().shouldEmitFloat16WithExcessPrecision()) {
    if (Ty->isAnyComplexType()) {
      QualType ElementType = Ty->castAs<clang::ComplexType>()->getElementType();
      if (ElementType->isFloat16Type())
        return CGM.getContext().getComplexType(CGM.getContext().FloatTy);
    }
    if (Ty->isFloat16Type())
      return CGM.getContext().FloatTy;
  }
  return QualType();
}

} // namespace CodeGen
} // namespace mlirclang
