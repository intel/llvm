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

#include "clang/AST/ASTContext.h"
#include "clang/CodeGen/CGFunctionInfo.h"

#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/../../lib/CodeGen/TargetInfo.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cgeist"

using namespace clang;
using namespace mlir;
using namespace llvm;

static cl::opt<bool>
    memRefFullRank("memref-fullrank", cl::init(false),
                   cl::desc("Get the full rank of the memref."));

static cl::opt<bool> memRefABI("memref-abi", cl::init(true),
                               cl::desc("Use memrefs when possible"));

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
operator<<(llvm::raw_ostream &os, clang::CodeGen::ABIArgInfo::Kind &ArgInfo) {
  os << "ABIArgInfo::";
  switch (ArgInfo) {
  case clang::CodeGen::ABIArgInfo::Direct:
    return os << "Direct";
  case clang::CodeGen::ABIArgInfo::Extend:
    return os << "Extend";
  case clang::CodeGen::ABIArgInfo::Indirect:
    return os << "Indirect";
  case clang::CodeGen::ABIArgInfo::IndirectAliased:
    return os << "IndirectAliased";
  case clang::CodeGen::ABIArgInfo::Ignore:
    return os << "Ignore";
  case clang::CodeGen::ABIArgInfo::Expand:
    return os << "Expand";
  case clang::CodeGen::ABIArgInfo::CoerceAndExpand:
    return os << "CoerceAndExpand";
  case clang::CodeGen::ABIArgInfo::InAlloca:
    return os << "InAlloca";
  }
  llvm_unreachable("Invalid ABI kind");
  return os;
}

/// Iteratively get the size of each dim of the given ConstantArrayType inst.
static void
getConstantArrayShapeAndElemType(const clang::QualType &ty,
                                 llvm::SmallVectorImpl<int64_t> &shape,
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

      if (AI.isDirect() && AI.getCanBeFlattened() && STy)
        llvm::errs() << "Warning: struct should be flattened but MLIR codegen "
                        "cannot yet handle it. Needs to be fixed.";

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
      TheCXXABI(CGM.getCXXABI()),
      TheABIInfo(CGM.getTargetCodeGenInfo().getABIInfo()) {}

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
  auto getDeclArgTy = [&](int32_t ArgNo, QualType ABIArgTy) {
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
  auto getMLIRArgType = [this](QualType QT) {
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
      unsigned AddressSpace = CGM.getContext().getTargetAddressSpace(Ret);
      ResultType = getPointerOrMemRefType(Ty, AddressSpace);
    } else {
      ResultType = Builder.getNoneType();
    }
    break;

  case clang::CodeGen::ABIArgInfo::Indirect:
    if (!AllowSRet) {
      // HACK: remove once we can handle function returning a struct.
      llvm::errs() << "Warning: function should return its value indirectly "
                      "(as an extra reference parameter). This is not yet "
                      "handled by the MLIR codegen\n";
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
    unsigned AddressSpace = CGM.getContext().getTargetAddressSpace(Ret);
    ArgTypes[IRFunctionArgs.getSRetArgNo()] =
        mlir::MemRefType::get(-1, Ty, {}, AddressSpace);
  }

  // Add type for inalloca argument.
  if (AllowInAllocaRet && IRFunctionArgs.hasInallocaArg()) {
    llvm_unreachable("not implemented");
    auto ArgStruct = FI.getArgStruct();
    assert(ArgStruct);
    //  auto Ty = LLVM::LLVMStructType::getLiteral(TheModule->getContext(),
    //  ArgTys);
    // ArgTypes[IRFunctionArgs.getInallocaArgNo()] =
    //  mlir::MemRefType::get(-1, Ty, {}, 0);
  }

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  clang::CodeGen::CGFunctionInfo::const_arg_iterator
      it = FI.arg_begin(),
      ie = it + FI.getNumRequiredArgs();
  for (; it != ie; ++it, ++ArgNo) {
    // Note: 'ArgTy' is the type of the parameter after it as been decayed to
    // abide to the ABI rules.
    const QualType &ArgTy = it->type;
    const clang::CodeGen::ABIArgInfo &ArgInfo = it->info;

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
    const QualType DeclArgTy = getDeclArgTy(ArgNo, ArgTy);

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
      mlir::Type MLIRArgTy = getMLIRArgType(DeclArgTy);
      ArgTypes[FirstIRArg] = getPointerOrMemRefType(
          MLIRArgTy, CGM.getDataLayout().getAllocaAddrSpace());
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "mlir type: " << ArgTypes[FirstIRArg] << "\n");
      break;
    }
    case clang::CodeGen::ABIArgInfo::IndirectAliased: {
      assert(NumIRArgs == 1);
      mlir::Type MLIRArgTy = getMLIRArgType(DeclArgTy);
      ArgTypes[FirstIRArg] =
          getPointerOrMemRefType(MLIRArgTy, ArgInfo.getIndirectAddrSpace());
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "mlir type: " << ArgTypes[FirstIRArg] << "\n");
      break;
    }
    case clang::CodeGen::ABIArgInfo::Extend:
    case clang::CodeGen::ABIArgInfo::Direct: {
      mlir::Type MLIRArgTy = getMLIRArgType(DeclArgTy);

      // Fast-isel and the optimizer generally like
      // scalar values better than FCAs, so we flatten them if this is safe to
      // do for this argument.
      auto st = MLIRArgTy.dyn_cast<mlir::LLVM::LLVMStructType>();

      if (st && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened())
        llvm::errs() << "Warning: struct should be flattened but MLIR codegen "
                        "cannot yet handle it. Needs to be fixed.";

      if (AllowStructFlattening && st && ArgInfo.isDirect() &&
          ArgInfo.getCanBeFlattened()) {
        assert(NumIRArgs == st.getBody().size());
        for (unsigned i = 0, e = st.getBody().size(); i != e; ++i)
          ArgTypes[FirstIRArg + i] = st.getBody()[i];
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

mlir::Type CodeGenTypes::getMLIRType(clang::QualType qt, bool *implicitRef,
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
      // prevents us from doing more comprehensive analysis.Here we
      // specifically handle this case by unwrapping the clang-adjusted
      // type, to get the corresponding ConstantArrayType with the full
      // dimensions.
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
    return mlir::LLVM::LLVMStructType::getLiteral(TheModule->getContext(),
                                                  types);
  }

  mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(*TheModule->getContext());

  if (auto RT = dyn_cast<clang::RecordType>(qt)) {
    if (RT->getDecl()->isInvalidDecl()) {
      RT->getDecl()->dump();
      RT->dump();
    }
    assert(!RT->getDecl()->isInvalidDecl());
    if (TypeCache.find(RT) != TypeCache.end())
      return TypeCache[RT];
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
      if (TypeName == "range" || TypeName == "nd_range" ||
          TypeName == "array" || TypeName == "id" ||
          TypeName == "accessor_common" || TypeName == "accessor" ||
          TypeName == "AccessorImplDevice" || TypeName == "item" ||
          TypeName == "ItemBase" || TypeName == "nd_item" ||
          TypeName == "group") {
        return getSYCLType(RT, *this);
      }
      // No need special handling for types that don't have record declaration
      // name.
      if (TypeName != "")
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
      TypeCache[RT] = LLVM::LLVMStructType::getIdentified(
          TheModule->getContext(), ("polygeist@mlir@" + ST->getName()).str());

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
      innerSYCL |= mlir::sycl::isSYCLType(ty);
      types.push_back(ty);
    }

    if (types.empty())
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return typeTranslator.translateType(anonymize(ST));

    if (recursive) {
      auto LR = TypeCache[RT].setBody(types, /*isPacked*/ false);
      assert(LR.succeeded());
      return TypeCache[RT];
    }

    if (!memRefABI || notAllSame || !allowMerge || innerLLVM || innerSYCL)
      return LLVM::LLVMStructType::getLiteral(TheModule->getContext(), types);

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
    mlir::OpBuilder builder(TheModule->getContext());
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
      if (auto ST = subType.dyn_cast<mlir::LLVM::LLVMStructType>())
        InnerSYCL |= any_of(ST.getBody(), mlir::sycl::isSYCLType);

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
      OpBuilder builder(TheModule->getContext());
      return builder.getIntegerType(8);
    }
    llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
    mlir::OpBuilder builder(TheModule->getContext());
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

// Note: In principle we should always create a memref here because we want to
// avoid lowering the abstraction level at this point in the compilation flow.
// However, cgeist treats type inconsistently, it expects memref for SYCL types
// and pointers for every other struct type.
mlir::Type CodeGenTypes::getPointerOrMemRefType(mlir::Type Ty,
                                                unsigned AddressSpace,
                                                bool IsAlloc) {
  bool IsSYCLType = mlir::sycl::isSYCLType(Ty);
  if (auto ST = Ty.dyn_cast<mlir::LLVM::LLVMStructType>())
    IsSYCLType |= any_of(ST.getBody(), mlir::sycl::isSYCLType);

  if (IsSYCLType)
    return mlir::MemRefType::get(IsAlloc ? 1 : -1, Ty, {}, AddressSpace);

  return LLVM::LLVMPointerType::get(Ty, AddressSpace);
}

const clang::CodeGen::CGFunctionInfo &
CodeGenTypes::arrangeGlobalDeclaration(clang::GlobalDecl GD) {
  return CGM.getTypes().arrangeGlobalDeclaration(GD);
}

} // namespace CodeGen
} // namespace mlirclang
