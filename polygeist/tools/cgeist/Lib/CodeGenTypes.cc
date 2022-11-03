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
#include "llvm/Support/WithColor.h"
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
    auto CstArrTy = cast<clang::ConstantArrayType>(CurTy);
    Shape.push_back(CstArrTy->getSize().getSExtValue());
    CurTy = CstArrTy->getElementType();
  }

  ElemTy = CurTy;
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
        llvm::WithColor::warning()
            << "struct should be flattened but MLIR codegen "
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
      unsigned AddressSpace = CGM.getContext().getTargetAddressSpace(Ret);
      ResultType = getPointerOrMemRefType(Ty, AddressSpace);
    } else {
      ResultType = Builder.getNoneType();
    }
    break;

  case clang::CodeGen::ABIArgInfo::Indirect:
    if (!AllowSRet) {
      // HACK: remove once we can handle function returning a struct.
      llvm::WithColor::warning()
          << "function should return its value indirectly "
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
        getPointerOrMemRefType(Ty, AddressSpace);
  }

  // Add type for inalloca argument.
  if (AllowInAllocaRet && IRFunctionArgs.hasInallocaArg()) {
    llvm_unreachable("not implemented");
    auto ArgStruct = FI.getArgStruct();
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

      if (ST && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened())
        llvm::WithColor::warning()
            << "struct should be flattened but MLIR codegen "
               "cannot yet handle it. Needs to be fixed.";

      if (AllowStructFlattening && ST && ArgInfo.isDirect() &&
          ArgInfo.getCanBeFlattened()) {
        assert(NumIRArgs == ST.getBody().size());
        for (unsigned i = 0, e = ST.getBody().size(); i != e; ++i)
          ArgTypes[FirstIRArg + i] = ST.getBody()[i];
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
    bool AttrOnCallSite, bool IsThunk) {
  MLIRContext *Ctx = TheModule->getContext();
  mlirclang::AttrBuilder FuncAttrsBuilder(*Ctx);
  mlirclang::AttrBuilder RetAttrsBuilder(*Ctx);

  unsigned CC = FI.getEffectiveCallingConvention();
  FuncAttrsBuilder.addAttribute(
      "llvm.cconv", mlir::LLVM::CConvAttr::get(
                        Ctx, static_cast<mlir::LLVM::cconv::CConv>(CC)));

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
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::ReadNone);
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'const' functions have greater restrictions than
      // 'pure' functions, so they also cannot have infinite loops.
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::WillReturn);
    } else if (TargetDecl->hasAttr<PureAttr>()) {
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::ReadOnly);
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'pure' functions cannot have infinite loops.
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::WillReturn);
    } else if (TargetDecl->hasAttr<NoAliasAttr>()) {
      FuncAttrsBuilder.addPassThroughAttribute(llvm::Attribute::ArgMemOnly);
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
      Optional<unsigned> NumElemsParam;
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
        CodeGenUtils::determineNoUndef(RetTy, CGM.getTypes(), DL, RetAI))
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
    // inalloca and sret disable readnone and readonly
    FuncAttrsBuilder.removeAttribute(llvm::Attribute::ReadOnly)
        .removeAttribute(llvm::Attribute::ReadNone);
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
      if (CGM.getContext().getTargetAddressSpace(PTy) == 0 &&
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
    ArgAttrs[IRFunctionArgs.getSRetArgNo()] = SRETAttrs.getAttrs();
  }

  // Attach attributes to inalloca argument.
  if (IRFunctionArgs.hasInallocaArg()) {
    mlirclang::AttrBuilder Attrs(*Ctx);
    assert(false && "TODO");
    //    Attrs.addAttribute(llvm::Attribute::InAlloca,
    //                     getMLIRType(FI.getArgStruct()));
    ArgAttrs[IRFunctionArgs.getInallocaArgNo()] = Attrs.getAttrs();
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

    if (!CGM.getCodeGenOpts().NullPointerIsValid &&
        CGM.getContext().getTargetAddressSpace(FI.arg_begin()->type) == 0) {
      ParamAttrsBuilder.addAttribute(llvm::Attribute::NonNull);
      ParamAttrsBuilder.addPassThroughAttribute(
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
    ArgAttrs[IRArgs.first] = ParamAttrsBuilder.getAttrs();
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
                .getAttrs();
      }
    }

    // Decide whether the argument we're handling could be partially undef
    if (CGM.getCodeGenOpts().EnableNoundefAttrs &&
        CodeGenUtils::determineNoUndef(ParamType, CGM.getTypes(), DL, AI)) {
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
      FuncAttrsBuilder.removeAttribute(llvm::Attribute::ReadOnly)
          .removeAttribute(llvm::Attribute::ReadNone);

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
      // inalloca disables readnone and readonly.
      FuncAttrsBuilder.removeAttribute(llvm::Attribute::ReadOnly)
          .removeAttribute(llvm::Attribute::ReadNone);
      continue;
    }

    if (const auto *RefTy = ParamType->getAs<ReferenceType>()) {
      QualType PTy = RefTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        ParamAttrsBuilder.addAttribute(
            llvm::Attribute::Dereferenceable,
            CGM.getMinimumObjectSize(PTy).getQuantity());
      if (CGM.getContext().getTargetAddressSpace(PTy) == 0 &&
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
      for (unsigned i = 0; i < NumIRArgs; i++)
        ArgAttrs[FirstIRArg + i].append(ParamAttrsBuilder.getAttrs());
    }
  }
  assert(ArgNo == FI.arg_size());

  AttrList.addAttrs(FuncAttrsBuilder, RetAttrsBuilder, ArgAttrs);
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

  mlir::LLVM::TypeFromLLVMIRTranslator TypeTranslator(*TheModule->getContext());

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
        llvm::WithColor::warning() << "SYCL type '" << ST->getName()
                                   << "' has not been converted to SYCL MLIR\n";
    }

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (CodeGenUtils::isLLVMStructABI(RT->getDecl(), ST))
      return TypeTranslator.translateType(anonymize(ST));

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
        return TypeTranslator.translateType(anonymize(ST));

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
      return TypeTranslator.translateType(T);
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
      return TypeTranslator.translateType(T);
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
    if (T->is16bitFPTy()) {
      if (CGM.getTarget().shouldEmitFloat16WithExcessPrecision()) {
        llvm::WithColor::warning()
            << "Experimental usage of _Float16. Code generated will be illegal "
               "for this target. Use with caution.\n";
      }
      return builder.getF16Type();
    }

    if (auto IT = dyn_cast<llvm::IntegerType>(T)) {
      return builder.getIntegerType(IT->getBitWidth());
    }
  }
  qt->dump();
  assert(0 && "unhandled type");
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
    return mlir::MemRefType::get(IsAlloc ? 1 : -1, Ty, {}, AddressSpace);

  return LLVM::LLVMPointerType::get(Ty, AddressSpace);
}

const clang::CodeGen::CGFunctionInfo &
CodeGenTypes::arrangeGlobalDeclaration(clang::GlobalDecl GD) {
  return CGM.getTypes().arrangeGlobalDeclaration(GD);
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

} // namespace CodeGen
} // namespace mlirclang
