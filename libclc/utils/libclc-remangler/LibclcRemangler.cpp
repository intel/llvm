//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for ensuring consistent mangling between libclc
// and the target. This tool remangles all functions using `long`,
// `unsigned long`, and `char` to appear as if they use `long long`,
// `unsigned long long`, and `signed char`, as is consistent with the primitive
// types defined by OpenCL C. Following a remangling, the original function
// mangling will be built as a clone of either the remangled function or a
// function with a suitable function if any exists. In some cases a clone of
// the remangled function is created for functions where multiple parameters
// have been replaced, and the replaced values are aliases.
//
// Original Clone Example:
//          If libclc defined a function `f(long)` the mangled name would be
//          `_Z1fl`. The remangler would rename this function to `_Z1fx`
//          (`f(long long)`.) If the target uses 64-bit `long`, `_Z1fl` is
//          cloned from the old function now under the name `_Z1fx`, whereas if
//          the target uses 32-bit `long`, `_Z1fl` is cloned from `_Z1fi`
//          (`f(int)`) if such a function exists.
//
// Remangled Clone Example:
//          In cases where the remangled name squashes valid versions of a
//          function a clone is created. `f(long, char, signed char)` would be
//          mangled to `_Z1flca`. The remangler would rename this function to
//          `_Z1fxaa` (`f(long long, signed char, signed char)`). If the target
//          uses a signed char then a valid clone `_Z1fxca`, (`f(long long,
//          char, signed char)`), is not defined. The remangler creates a clone
//          of the renamed function,`_Z1fxaa`, to this permutation, `_Z1fxca`.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Mangle.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <iostream>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace llvm::itanium_demangle;

enum class Signedness { Signed, Unsigned };
enum class SupportedLongWidth { L32, L64 };

static ExitOnError ExitOnErr;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory
    LibCLCRemanglerToolCategory("libclc-remangler-tool options");

static cl::opt<std::string>
    InputIRFilename("input-ir", cl::desc("<input bitcode>"),
                    cl::cat(LibCLCRemanglerToolCategory));
static cl::opt<std::string> OutputFilename("o", cl::init("-"),
                                           cl::desc("Output filename"));
static cl::opt<SupportedLongWidth>
    LongWidth("long-width",
              cl::values(clEnumValN(SupportedLongWidth::L32, "l32",
                                    "long is 32-bit wide."),
                         clEnumValN(SupportedLongWidth::L64, "l64",
                                    "long is 64-bit wide.")),
              cl::cat(LibCLCRemanglerToolCategory));
static cl::opt<Signedness> CharSignedness(
    "char-signedness",
    cl::values(clEnumValN(Signedness::Signed, "signed", "char is signed."),
               clEnumValN(Signedness::Unsigned, "unsigned",
                          "char is unsigned.")),
    cl::cat(LibCLCRemanglerToolCategory));
static cl::opt<bool> Verbose("v", cl::desc("Enable verbose output"),
                             cl::init(false),
                             cl::cat(LibCLCRemanglerToolCategory));
static cl::opt<bool> TextualOut("S", cl::desc("Emit LLVM textual assembly"),
                                cl::init(false),
                                cl::cat(LibCLCRemanglerToolCategory));
static cl::opt<bool> TestRun("t", cl::desc("Enable test run"), cl::init(false),
                             cl::cat(LibCLCRemanglerToolCategory));

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

namespace {
inline StringRef asRef(std::string_view S) { return {&*S.begin(), S.size()}; }
class BumpPointerAllocator {
public:
  BumpPointerAllocator()
      : BlockList(new(InitialBuffer) BlockMeta{nullptr, 0}) {}

  void *allocate(size_t N) {
    N = (N + 15u) & ~15u;
    if (N + BlockList->Current >= UsableAllocSize) {
      if (N > UsableAllocSize)
        return allocateMassive(N);
      grow();
    }
    BlockList->Current += N;
    return static_cast<void *>(reinterpret_cast<char *>(BlockList + 1) +
                               BlockList->Current - N);
  }

  void reset() {
    while (BlockList) {
      BlockMeta *Tmp = BlockList;
      BlockList = BlockList->Next;
      if (reinterpret_cast<char *>(Tmp) != InitialBuffer)
        std::free(Tmp);
    }
    BlockList = new (InitialBuffer) BlockMeta{nullptr, 0};
  }

  ~BumpPointerAllocator() { reset(); }

private:
  void grow() {
    char *NewMeta = static_cast<char *>(std::malloc(AllocSize));
    if (NewMeta == nullptr)
      std::terminate();
    BlockList = new (NewMeta) BlockMeta{BlockList, 0};
  }

  void *allocateMassive(size_t NBytes) {
    NBytes += sizeof(BlockMeta);
    BlockMeta *NewMeta = reinterpret_cast<BlockMeta *>(std::malloc(NBytes));
    if (NewMeta == nullptr)
      std::terminate();
    BlockList->Next = new (NewMeta) BlockMeta{BlockList->Next, 0};
    return static_cast<void *>(NewMeta + 1);
  }

private:
  struct BlockMeta {
    BlockMeta *Next;
    size_t Current;
  };

  static constexpr size_t AllocSize = 4096;
  static constexpr size_t UsableAllocSize = AllocSize - sizeof(BlockMeta);

  alignas(max_align_t) char InitialBuffer[AllocSize];
  BlockMeta *BlockList = nullptr;
};

class DefaultAllocator {
public:
  void reset() { Alloc.reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&...A) {
    return new (Alloc.allocate(sizeof(T))) T(std::forward<Args>(A)...);
  }

  void *allocateNodeArray(size_t Sz) {
    return Alloc.allocate(sizeof(itanium_demangle::Node *) * Sz);
  }

private:
  BumpPointerAllocator Alloc;
};

clang::QualType getBaseType(StringRef Name, clang::ASTContext *AST,
                            bool &IsVariadic) {
  clang::QualType Res;
  // First find the match against `QualType`...
  if (Name == "void")
    Res = AST->VoidTy;
  else if (Name == "wchar_t")
    Res = AST->WCharTy;
  else if (Name == "bool")
    Res = AST->BoolTy;
  else if (Name == "char")
    Res = AST->CharTy;
  else if (Name == "signed char")
    Res = AST->SignedCharTy;
  else if (Name == "unsigned char")
    Res = AST->UnsignedCharTy;
  else if (Name == "short")
    Res = AST->ShortTy;
  else if (Name == "unsigned short")
    Res = AST->UnsignedShortTy;
  else if (Name == "int")
    Res = AST->IntTy;
  else if (Name == "unsigned int")
    Res = AST->UnsignedIntTy;
  else if (Name == "long")
    Res = AST->LongTy;
  else if (Name == "unsigned long")
    Res = AST->UnsignedLongTy;
  else if (Name == "long long")
    Res = AST->LongLongTy;
  else if (Name == "unsigned long long")
    Res = AST->UnsignedLongLongTy;
  else if (Name == "__int128")
    Res = AST->Int128Ty;
  else if (Name == "unsigned __int128")
    Res = AST->UnsignedInt128Ty;
  else if (Name == "float")
    Res = AST->FloatTy;
  else if (Name == "double")
    Res = AST->DoubleTy;
  else if (Name == "long double")
    Res = AST->LongDoubleTy;
  else if (Name == "__float128")
    Res = AST->Float128Ty;
  else if (Name == "...") {
    Res = clang::QualType{};
    IsVariadic = true;
  } else if (Name == "decimal64")
    assert(false && "unhandled type name: decimal64");
  else if (Name == "decimal128")
    assert(false && "unhandled type name: decimal128");
  else if (Name == "decimal32")
    assert(false && "unhandled type name: decimal32");
  else if (Name == "decimal16")
    assert(false && "unhandled type name: decimal16");
  else if (Name == "char32_t")
    Res = AST->Char32Ty;
  else if (Name == "char16_t")
    Res = AST->Char16Ty;
  else if (Name == "char8_t")
    Res = AST->Char8Ty;
  else if (Name == "_Float16")
    Res = AST->Float16Ty;
  else if (Name == "half")
    Res = AST->HalfTy;
  else if (Name == "auto")
    Res = AST->AutoDeductTy;
  else if (Name == "decltype(auto)")
    assert(false && "unhandled type name: decltype(auto)");
  else if (Name == "std::nullptr_t")
    Res = AST->NullPtrTy;
  else if (Name == "_BitInt")
    assert(false && "unhandled type name: _BitInt");
  else {
    auto &II = AST->Idents.get(Name);
    auto *DC = AST->getTranslationUnitDecl();
    auto *ED = EnumDecl::Create(*AST, DC, SourceLocation(), SourceLocation(),
                                &II, nullptr, false, false, true);
    Res = AST->getEnumType(ED);
  }
  return Res;
}
} // unnamed namespace

using Demangler = ManglingParser<DefaultAllocator>;

class Remangler {
public:
  Remangler(ASTContext *AST, const Node *Root,
            SmallDenseMap<const char *, const char *> TypeReplacements)
      : AST(AST), Root(Root), TypeReplacements(TypeReplacements) {
    MangleContext.reset(
        ItaniumMangleContext::create(*AST, AST->getDiagnostics()));
  }

  bool hasFailed() { return Failed; }

  // Generate mangled function name, based on a given itanium_demangle `Node`.
  std::string remangle() {
    clang::QualType RetTy;
    SmallVector<clang::QualType> TemplateArgTys;
    SmallVector<clang::QualType> InputArgTys;
    bool IsVariadic = false;
    nodeToQualTypes(RetTy, TemplateArgTys, InputArgTys, IsVariadic);
    auto *FD = createKernelDecl(RetTy, TemplateArgTys, InputArgTys, IsVariadic);
    assert(MangleContext->shouldMangleDeclName(FD) &&
           "It should always be possible to mangle libclc func.");

    std::string Buf;
    raw_string_ostream Out(Buf);
    MangleContext->mangleName(FD, Out);
    return Buf;
  }

private:
  // Helper struct to aggregate information about types.
  struct NodeKindInfo {
    NodeKindInfo(Node::Kind K) : K(K) {}
    NodeKindInfo(Node::Kind K, size_t NumElemsOrAS) : K(K) {
      Data = NumElemsOrAS;
    }
    NodeKindInfo(Node::Kind K, itanium_demangle::Qualifiers Quals) : K(K) {
      Data = 0;
      if (Quals & itanium_demangle::Qualifiers::QualConst)
        Data |= clang::Qualifiers::TQ::Const;
      if (Quals & itanium_demangle::Qualifiers::QualVolatile)
        Data |= clang::Qualifiers::TQ::Volatile;
      if (Quals & itanium_demangle::Qualifiers::QualRestrict)
        Data |= clang::Qualifiers::TQ::Restrict;
    }
    NodeKindInfo(Node::Kind K, const char *S, size_t N) : K(K) {
      DataStr.assign(S, N);
    }
    Node::Kind K;
    size_t Data = 0;
    std::string DataStr;
  };

  // Construct FunctionDecl from return, argument and template types.
  FunctionDecl *createKernelDecl(
      clang::QualType RetTy, const SmallVector<clang::QualType> &TemplateArgTys,
      const SmallVector<clang::QualType> &InputArgTys, bool IsVariadic) {
    // Copy in InputArgTys as this function can mutate them.
    auto ArgTys{InputArgTys};
    // Create this with a void ret no args prototype, will be fixed up after
    // we've seen all the params.
    FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
    Info.Variadic = IsVariadic;
    clang::QualType const VoidFuncType =
        AST->getFunctionType(AST->VoidTy, {}, Info);
    FunctionDecl *FD = FunctionDecl::Create(
        *AST, AST->getTranslationUnitDecl(), SourceLocation{},
        DeclarationNameInfo(), VoidFuncType,
        AST->getTrivialTypeSourceInfo(AST->VoidTy), SC_None, false, false,
        false, ConstexprSpecKind::Unspecified, nullptr);
    FD->setImplicitlyInline(false);

    // Set the name.
    const auto *Encoding = static_cast<const FunctionEncoding *>(Root);
    assert(
        (Encoding->getName()->getKind() == Node::Kind::KNameType ||
         Encoding->getName()->getKind() == Node::Kind::KNameWithTemplateArgs) &&
        "Expected KNameType or KNameWithTemplateArgs node.");
    StringRef KernelName;
    if (Encoding->getName()->getKind() == Node::Kind::KNameType) {
      auto *NT = static_cast<const NameType *>(Encoding->getName());
      KernelName = asRef(NT->getBaseName());
    } else {
      auto *NT = static_cast<const NameWithTemplateArgs *>(Encoding->getName());
      KernelName = asRef(NT->getBaseName());
    }
    FD->setDeclName(&AST->Idents.get(KernelName));

    // Construct the argument list.
    SmallVector<ParmVarDecl *, 8> ArgParams;
    for (auto &QT : ArgTys) {
      auto &II = AST->Idents.get("");
      auto *TTSI = AST->getTrivialTypeSourceInfo(QT);
      auto *NewParam =
          ParmVarDecl::Create(*AST, FD, SourceLocation(), SourceLocation(), &II,
                              QT, TTSI, SC_None, nullptr);
      NewParam->setScopeInfo(0, ArgParams.size());
      NewParam->setDeclContext(FD);
      ArgParams.push_back(NewParam);
    }

    // If not templated, finish here.
    if (TemplateArgTys.empty()) {
      clang::QualType const FuncType =
          AST->getFunctionType(RetTy, ArgTys, Info);
      FD->setType(FuncType);
      FD->setParams(ArgParams);
      return FD;
    }

    // Use FD as a base for a future function specialisation.
    FunctionDecl *FDSpecialization = FunctionDecl::Create(
        *AST, AST->getTranslationUnitDecl(), SourceLocation{},
        DeclarationNameInfo(), VoidFuncType,
        AST->getTrivialTypeSourceInfo(AST->VoidTy), SC_None, false, false,
        false, ConstexprSpecKind::Unspecified, nullptr);
    FDSpecialization->setImplicitlyInline(false);

    FDSpecialization->setDeclName(&AST->Idents.get(KernelName));

    // Will be used to build template parameter list.
    SmallVector<NamedDecl *> TemplateNamedDecls;
    // Used for setting template specialisation.
    SmallVector<TemplateArgument> TemplateArguments;
    // Used for the fixups.
    SmallVector<clang::QualType> TemplateTypeParamTys;
    unsigned TemplateIndex = 0;
    for (auto &TemplateArgQT : TemplateArgTys) {
      std::string const Name{std::string{"TempTy"} +
                             std::to_string(TemplateIndex)};
      auto &II = AST->Idents.get(Name);
      auto *TTPD = TemplateTypeParmDecl::Create(
          *AST, FDSpecialization->getDeclContext(), SourceLocation(),
          SourceLocation(), 0, TemplateIndex, &II, /* Typename */ true,
          /*ParameterPack*/ false);
      TTPD->setDefaultArgument(*AST,
        TemplateArgumentLoc());

      TemplateNamedDecls.emplace_back(TTPD);
      auto TA = TemplateArgument(TemplateArgQT);
      TemplateArguments.emplace_back(TA);

      // Store this qualified type with newly created proper template type
      // param qualified type.
      TemplateTypeParamTys.push_back(
          AST->getTemplateTypeParmType(0, TemplateIndex, false, TTPD));

      ++TemplateIndex;
    }
    // Fix up the template types in the original FD's arg tys and return ty.
    auto AreQTsEqual = [&](const clang::QualType &LHS,
                           const clang::QualType &RHS) -> bool {
      auto *LID = LHS.getBaseTypeIdentifier();
      auto *RID = RHS.getBaseTypeIdentifier();
      return (RID && LID && LID->isStr(RID->getName())) || LHS == RHS;
    };
    unsigned NumReplaced = 0;
    unsigned Idx = 0;
    for (auto &TemplateArgQT : TemplateArgTys) {
      if (AreQTsEqual(TemplateArgQT, RetTy)) {
        RetTy = TemplateTypeParamTys[Idx];
        goto Found;
      }
      for (unsigned i = 0; i < ArgTys.size(); ++i) {
        if (AreQTsEqual(ArgTys[i], TemplateArgQT)) {
          ArgTys[i] = TemplateTypeParamTys[Idx];
          goto Found;
        }
      }
    Found:
      ++NumReplaced;
      ++Idx;
    }
    assert(NumReplaced >= TemplateTypeParamTys.size() &&
           "Expected full specialization.");
    // Now that the template types have been patched up, set functions type.
    clang::QualType const TemplateFuncType =
        AST->getFunctionType(RetTy, ArgTys, Info);
    FD->setType(TemplateFuncType);
    FD->setParams(ArgParams);
    FDSpecialization->setType(TemplateFuncType);
    FDSpecialization->setParams(ArgParams);

    auto *TPL = TemplateParameterList::Create(
        *AST, SourceLocation(), SourceLocation(), TemplateNamedDecls,
        SourceLocation(), nullptr);
    auto *FTD = FunctionTemplateDecl::Create(*AST, FD->getDeclContext(),
                                             SourceLocation(),
                                             DeclarationName(), TPL, FD);
    auto TAArr = ArrayRef(TemplateArguments.begin(), TemplateArguments.size());
    auto *TAL = TemplateArgumentList::CreateCopy(*AST, TAArr);
    FDSpecialization->setTemplateParameterListsInfo(*AST, TPL);
    FDSpecialization->setFunctionTemplateSpecialization(
        FTD, TAL, nullptr, TSK_ExplicitSpecialization);

    return FDSpecialization;
  }

  // Peel off additional type info, such as CV qualifiers or pointers, by
  // recursively calling itself. The information is appended to `PossibleKinds`
  // vector.
  // The base case is achieved in `handleLeafTypeNode`.
  std::pair<clang::QualType, bool>
  handleTypeNode(const Node *TypeNode,
                 SmallVector<Remangler::NodeKindInfo> &PossibleKinds) {
    auto Kind = TypeNode->getKind();
    switch (Kind) {
    case Node::Kind::KPointerType: {
      PossibleKinds.push_back(NodeKindInfo(Kind));
      const auto *PType =
          static_cast<const itanium_demangle::PointerType *>(TypeNode);
      return handleTypeNode(PType->getPointee(), PossibleKinds);
    }
    case Node::Kind::KVectorType: {
      const auto *VecType =
          static_cast<const itanium_demangle::VectorType *>(TypeNode);
      assert(VecType->getDimension()->getKind() == Node::Kind::KNameType);
      const auto *Dims = static_cast<const itanium_demangle::NameType *>(
          VecType->getDimension());
      size_t DimNum;
      if (asRef(Dims->getName()).getAsInteger(10, DimNum) || !DimNum) {
        assert(false && "invalid vector size specifier");
        break;
      }
      PossibleKinds.push_back(NodeKindInfo(Kind, DimNum));
      return handleTypeNode(VecType->getBaseType(), PossibleKinds);
    }
    case Node::Kind::KBinaryFPType: {
      const itanium_demangle::BinaryFPType *BFPType =
          static_cast<const itanium_demangle::BinaryFPType *>(TypeNode);
      assert(BFPType->getDimension()->getKind() == Node::Kind::KNameType);
      const auto *NameTypeNode =
          static_cast<const itanium_demangle::NameType *>(
              BFPType->getDimension());
      assert(asRef(NameTypeNode->getBaseName()) == "16" &&
             "Unexpected binary floating point type.");
      // BinaryFPType is encoded as: BinaryFPType(NameType("16")), manually
      // construct "_Float16" NamedType node so we can pass it directly to
      // handleLeafTypeNode.
      NameType const FP16{"_Float16"};
      return handleLeafTypeNode(FP16.getName(), PossibleKinds);
    }
    case Node::Kind::KVendorExtQualType: {
      const auto *ExtQualType =
          static_cast<const itanium_demangle::VendorExtQualType *>(TypeNode);
      StringRef AS = asRef(ExtQualType->getExt());
      if (!AS.starts_with("AS")) {
        assert(false && "Unexpected ExtQualType.");
        break;
      }
      size_t ASNum;
      if (AS.drop_front(2).getAsInteger(10, ASNum)) {
        assert(false && "Unexpected ExtQualType.");
        break;
      }
      PossibleKinds.push_back({Kind, ASNum});
      return handleTypeNode(ExtQualType->getTy(), PossibleKinds);
    }
    case Node::Kind::KQualType: {
      auto *QType = static_cast<const itanium_demangle::QualType *>(TypeNode);
      PossibleKinds.push_back({Kind, QType->getQuals()});
      return handleTypeNode(QType->getChild(), PossibleKinds);
    }
    case Node::Kind::KNameType: {
      auto *NT = static_cast<const itanium_demangle::NameType *>(TypeNode);
      return handleLeafTypeNode(NT->getName(), PossibleKinds);
    }
    case Node::Kind::KNestedName: {
      const auto *NN =
          static_cast<const itanium_demangle::NestedName *>(TypeNode);
      OutputBuffer QB;
      NN->Qual->print(QB);
      PossibleKinds.push_back({Kind, QB.getBuffer(), QB.getCurrentPosition()});
      auto *NT = static_cast<const itanium_demangle::NameType *>(NN->Name);
      return handleLeafTypeNode(NT->getName(), PossibleKinds);
    }
    default: {
      OutputBuffer ErrorTypeOut;
      TypeNode->print(ErrorTypeOut);
      errs() << "Unhandled type: " << ErrorTypeOut.getBuffer() << "\n";
      free(ErrorTypeOut.getBuffer());
      Failed = true;
    }
    }

    llvm_unreachable("Unhandled type.");
    return std::make_pair(clang::QualType{}, false);
  }

  // Handle undecorated type that can be matched against `QualType`, also
  // returning if variadic.
  std::pair<clang::QualType, bool>
  handleLeafTypeNode(std::string_view Name,
                     SmallVector<NodeKindInfo> &PossibleKinds) {
    return handleLeafTypeNode(asRef(Name), PossibleKinds);
  }

  std::pair<clang::QualType, bool>
  handleLeafTypeNode(StringRef Name, SmallVector<NodeKindInfo> &PossibleKinds) {

    // When in test run, don't enable replacements and assert that re-mangled
    // name matches the original.
    if (!TestRun) {
      auto It = TypeReplacements.find(Name.begin());
      if (It != TypeReplacements.end())
        Name = It->second;
    }

    bool IsVariadic = false;
    clang::QualType Res = getBaseType(Name, AST, IsVariadic);

    // then apply gathered information to that `QualType`.

    // Handle `KNestedName` first, as it will create a new `QualType`.
    auto KNNMatcher = [](NodeKindInfo &NKI) {
      return NKI.K == Node::Kind::KNestedName;
    };
    auto *KNN =
        std::find_if(PossibleKinds.begin(), PossibleKinds.end(), KNNMatcher);
    if (KNN != PossibleKinds.end()) {
      assert(PossibleKinds.end() == std::find_if(std::next(KNN),
                                                 PossibleKinds.end(),
                                                 KNNMatcher) &&
             "Expected only one KNestedName kind.");

      // Construct the full name to check if it has already been handled.
      std::string const N{KNN->DataStr + " " +
                          Res.getBaseTypeIdentifier()->getName().str()};
      if (NestedNamesQTMap.count(N) == 0) {
        assert(StringRef(KNN->DataStr).starts_with("__spv") &&
               "Unexpected nested prefix");
        SourceLocation const SL{};
        RecordDecl *RD = nullptr;
        if (!SpvNamespace)
          SpvNamespace = NamespaceDecl::Create(
              *AST, AST->getTranslationUnitDecl(), false, SL, SL,
              &AST->Idents.get("__spv", tok::TokenKind::identifier), nullptr,
              false);
        std::string StructName =
            StringRef(KNN->DataStr).split("__spv::").second.str();
        auto *II = &AST->Idents.get(StructName, tok::TokenKind::identifier);
        RD = RecordDecl::Create(*AST, TagTypeKind::Struct, SpvNamespace, SL, SL, II);
        auto *NNS = NestedNameSpecifier::Create(*AST, nullptr, SpvNamespace);
        auto RecordQT = AST->getRecordType(RD);
        NNS = NestedNameSpecifier::Create(*AST, NNS, false,
                                          RecordQT.getTypePtr());
        auto &EnumName =
            AST->Idents.get(Res.getBaseTypeIdentifier()->getName());
        // We need to recreate the enum, now that we have access to all the
        // namespace/class info.
        auto *ED =
            EnumDecl::Create(*AST, RD, SourceLocation(), SourceLocation(),
                             &EnumName, nullptr, false, false, true);
        Res = AST->getEnumType(ED);
        Res = AST->getElaboratedType(ElaboratedTypeKeyword::None, NNS, Res);
        // Store the elaborated type for reuse, this is important as clang uses
        // substitutions for ET based on the object not the name enclosed in.
        NestedNamesQTMap[N] = Res;
      } else
        Res = NestedNamesQTMap[N];
    }

    // Iterate in reversed order to preserve the semantics.
    for (auto I = PossibleKinds.rbegin(); I != PossibleKinds.rend(); ++I) {
      switch (I->K) {
      case Node::Kind::KPointerType: {
        Res = AST->getPointerType(Res);
        break;
      }
      case Node::Kind::KVectorType: {
        Res = AST->getVectorType(Res, I->Data,
                                 clang::VectorKind::Generic);
        break;
      }
      case Node::Kind::KQualType: {
        auto Quals = clang::Qualifiers::fromFastMask(I->Data);
        Res = AST->getQualifiedType(Res, Quals);
        break;
      }
      case Node::Kind::KVendorExtQualType: {
        auto AS = getLangASFromTargetAS(I->Data);
        Res = AST->getAddrSpaceQualType(Res, AS);
        break;
      }
      case Node::Kind::KNestedName: {
        // Handled already.
        break;
      }
      default: {
        llvm_unreachable("Unexpected Node Kind.");
      }
      }
    }

    return std::make_pair(Res, IsVariadic);
  }

  // Traverse the itanium_demangle node and generate QualTypes corresponding to
  // the function's return type, input arguments and template params.
  void nodeToQualTypes(clang::QualType &RetTy,
                       SmallVector<clang::QualType> &TemplateArgTys,
                       SmallVector<clang::QualType> &ArgTys, bool &IsVariadic) {
    const FunctionEncoding *Encoding =
        static_cast<const FunctionEncoding *>(Root);

    SmallVector<NodeKindInfo> PossibleKinds;
    if (Encoding->getReturnType()) {
      RetTy =
          std::get<0>(handleTypeNode(Encoding->getReturnType(), PossibleKinds));
    } else
      RetTy = AST->VoidTy;

    if (Encoding->getName()->getKind() == Node::Kind::KNameWithTemplateArgs) {
      const NameWithTemplateArgs *NWTA =
          static_cast<const NameWithTemplateArgs *>(Encoding->getName());
      assert(NWTA->getKind() == Node::Kind::KNameWithTemplateArgs);
      const TemplateArgs *TA =
          static_cast<const class TemplateArgs *>(NWTA->TemplateArgs);
      for (auto *TPT : TA->getParams()) {
        PossibleKinds.clear();
        auto Res = handleTypeNode(TPT, PossibleKinds);
        assert(!std::get<1>(Res) && "Variadic in template params.");
        TemplateArgTys.push_back(std::get<0>(Res));
      }
    }

    for (auto *PT : Encoding->getParams()) {
      PossibleKinds.clear();
      auto Res = handleTypeNode(PT, PossibleKinds);
      if (std::get<1>(Res)) {
        IsVariadic = true;
        continue;
      }
      ArgTys.push_back(std::get<0>(Res));
    }
  }

private:
  ASTContext *AST = nullptr;
  std::unique_ptr<clang::MangleContext> MangleContext{};
  const Node *Root = nullptr;
  SmallDenseMap<const char *, const char *> TypeReplacements{};

  bool Failed = false;

  std::map<std::string, clang::QualType> NestedNamesQTMap{};
  NamespaceDecl *SpvNamespace = nullptr;
};

class TargetTypeReplacements {
  SmallDenseMap<const char *, const char *> ParameterTypeReplacements;
  SmallDenseMap<const char *, const char *> CloneTypeReplacements;
  SmallDenseMap<const char *, const char *> RemangledCloneTypeReplacements;

  void createRemangledTypeReplacements() {
    // RemangleTypes which are not aliases or not the exact same alias
    // type
    for (auto &PTR : ParameterTypeReplacements) {
      const char *From = PTR.getFirst();
      const char *To = PTR.getSecond();
      if (CloneTypeReplacements.find(From) == CloneTypeReplacements.end())
        RemangledCloneTypeReplacements[From] = To;
      else if (CloneTypeReplacements[From] != To)
        RemangledCloneTypeReplacements[From] = To;
    }
  }

public:
  TargetTypeReplacements() {
    // Replace long with long long
    ParameterTypeReplacements["long"] = "long long";
    ParameterTypeReplacements["unsigned long"] = "unsigned long long";

    // Replace char with signed char
    ParameterTypeReplacements["char"] = "signed char";

    // Make replaced long functions clones of either integer or long
    // long variant
    if (LongWidth == SupportedLongWidth::L32) {
      CloneTypeReplacements["long"] = "int";
      CloneTypeReplacements["unsigned long"] = "unsigned int";
    } else {
      CloneTypeReplacements["long"] = "long long";
      CloneTypeReplacements["unsigned long"] = "unsigned long long";
    }

    // Make replaced char functions clones of explicit signed char or unsigned
    // char type
    if (CharSignedness == Signedness::Signed) {
      CloneTypeReplacements["char"] = "signed char";
    } else {
      CloneTypeReplacements["char"] = "unsigned char";
    }

    createRemangledTypeReplacements();
  }

  SmallDenseMap<const char *, const char *> getParameterTypeReplacements() {
    return ParameterTypeReplacements;
  }

  SmallDenseMap<const char *, const char *> getCloneTypeReplacements() {
    return CloneTypeReplacements;
  }

  SmallDenseMap<const char *, const char *>
  getRemangledCloneTypeReplacements() {
    return RemangledCloneTypeReplacements;
  }
};

class LibCLCRemangler : public ASTConsumer {
public:
  LibCLCRemangler() : ASTCtx(nullptr), LLVMCtx(), Replacements() {}

  void Initialize(ASTContext &C) override {
    ASTCtx = &C;
    std::unique_ptr<MemoryBuffer> const Buff = ExitOnErr(
        errorOrToExpected(MemoryBuffer::getFileOrSTDIN(InputIRFilename)));

    SMDiagnostic Err;
    std::unique_ptr<llvm::Module> const M =
        parseIR(Buff.get()->getMemBufferRef(), Err, LLVMCtx);

    if (!M) {
      Err.print("libclc-remangler", errs());
      exit(1);
    }

    handleModule(M.get());
  }

private:
  bool createClones(llvm::Module *M, StringRef OriginalMangledName,
                    std::string RemangledName,
                    const itanium_demangle::Node *FunctionTree,
                    TargetTypeReplacements &Replacements) {
    // create clone of original function
    if (!createCloneFromMap(M, OriginalMangledName, FunctionTree,
                            Replacements.getCloneTypeReplacements(),
                            /* CloneeTypeReplacement= */ true))
      return false;

    // create clone of remangled function
    return createCloneFromMap(M, RemangledName, FunctionTree,
                              Replacements.getRemangledCloneTypeReplacements());
  }

  bool
  createCloneFromMap(llvm::Module *M, StringRef OriginalName,
                     const itanium_demangle::Node *FunctionTree,
                     SmallDenseMap<const char *, const char *> TypeReplacements,
                     bool CloneeTypeReplacement = false) {
    Remangler ATR{ASTCtx, FunctionTree, TypeReplacements};

    std::string const RemangledName = ATR.remangle();

    if (ATR.hasFailed())
      return false;

    // Name has not changed from the original name.
    if (RemangledName == OriginalName)
      return true;

    StringRef CloneName, CloneeName;
    if (CloneeTypeReplacement) {
      CloneName = OriginalName;
      CloneeName = RemangledName;
    } else {
      CloneName = RemangledName;
      CloneeName = OriginalName;
    }

    if (Function *Clonee = M->getFunction(CloneeName)) {
      ValueToValueMapTy Dummy;
      Function *NewF = CloneFunction(Clonee, Dummy);
      NewF->setName(CloneName.str());
    } else if (Verbose) {
      errs() << "Could not create copy " << CloneName.data() << " : missing "
             << CloneeName.data() << '\n';
    }
    return true;
  }

  bool remangleFunction(Function &Func, llvm::Module *M) {
    if (!Func.getName().starts_with("_Z"))
      return true;

    std::string const MangledName = Func.getName().str();
    Demangler D{MangledName.data(), MangledName.data() + MangledName.size()};
    const itanium_demangle::Node *FunctionTree = D.parse();
    if (!FunctionTree) {
      errs() << "Unable to demangle name: " << MangledName << '\n';
      return false;
    }

    // Try to change the parameter types in the function name using the
    // mappings.
    Remangler R{ASTCtx, FunctionTree,
                Replacements.getParameterTypeReplacements()};

    std::string const RemangledName = R.remangle();

    if (R.hasFailed())
      return false;

    if (RemangledName != MangledName) {
      if (Verbose || TestRun) {
        errs() << "Mangling changed:"
               << "\n"
               << "Original:  " << MangledName << "\n"
               << "New:       " << RemangledName << "\n";
      }
      // In test run mode, where no substitution is made, change in mangling
      // name represents a failure. Report an error.
      if (TestRun) {
        errs() << "Test run failure!\n";
        return false;
      }
      Func.setName(RemangledName);

      // Make a clone of a suitable function using the old name if there is a
      // type-mapping and the corresponding clonee function exists.
      if (!createClones(M, MangledName, RemangledName, FunctionTree,
                        Replacements))
        return false;
    }
    return true;
  }

  void handleModule(llvm::Module *M) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> Out(
        new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
    if (EC) {
      errs() << EC.message() << '\n';
      exit(1);
    }

    // This module is built explicitly for linking with any .bc compiled with
    // the "nvptx64-nvidia-cuda" (CUDA) or "amdgcn-amd-amdhsa" (HIP AMD)
    // triples. Therefore we update the module triple.
    if (M->getTargetTriple() == "nvptx64-unknown-nvidiacl") {
      M->setTargetTriple("nvptx64-nvidia-cuda");
    } else if (M->getTargetTriple() == "amdgcn-unknown-amdhsa") {
      M->setTargetTriple("amdgcn-amd-amdhsa");
    }

    std::vector<Function *> FuncList;
    for (auto &Func : M->getFunctionList())
      FuncList.push_back(&Func);

    bool Success = true;
    for (auto *Func : FuncList)
      Success &= remangleFunction(*Func, M);
    // Only fail after all to give as much context as possible.
    if (!Success) {
      errs() << "Failed to remangle all mangled functions in module.\n";
      exit(1);
    }

    if (TestRun) {
      if (Verbose)
        errs() << "Successfully processed: " << FuncList.size()
               << " functions.\n";
      return;
    }

    if (TextualOut)
      M->print(Out->os(), nullptr, true);
    else
      WriteBitcodeToFile(*M, Out->os());

    // Declare success.
    Out->keep();
  }

private:
  ASTContext *ASTCtx;
  LLVMContext LLVMCtx;
  TargetTypeReplacements Replacements;
};

class LibCLCRemanglerActionFactory {
public:
  LibCLCRemanglerActionFactory() {}

  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return std::make_unique<LibCLCRemangler>();
  }
};

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(
      argc, argv, LibCLCRemanglerToolCategory, cl::ZeroOrMore);
  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    errs() << ExpectedParser.takeError();
    return 1;
  }

  // Use a default Compilation DB instead of the build one, as it might contain
  // toolchain specific options, not compatible with clang.
  FixedCompilationDatabase Compilations(".", std::vector<std::string>());
  ClangTool Tool(Compilations, ExpectedParser->getSourcePathList());

  LibCLCRemanglerActionFactory LRAF{};
  std::unique_ptr<FrontendActionFactory> FrontendFactory;
  FrontendFactory = newFrontendActionFactory(&LRAF);
  return Tool.run(FrontendFactory.get());
}
