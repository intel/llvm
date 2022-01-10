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
// mangling will be made an alias to either the remangled function or a function
// with a suitable function if any exists. In some cases an alias of the
// remangled function is created for functions where multiple parameters have
// been replaced, and the replaced values are aliases.
//
// Original Alias Example:
//          If libclc defined a function `f(long)` the mangled name would be
//          `_Z1fl`. The remangler would rename this function to `_Z1fx`
//          (`f(long long)`.) If the target uses 64-bit `long`, `_Z1fl` is made
//          an alias to the old function now under the name `_Z1fx`, whereas if
//          the target uses 32-bit `long`, `_Z1fl` is made an alias to `_Z1fi`
//          (`f(int)`) if such a function exists.
//
// Remangled Alias Example:
//          In cases where the remangled name squashes valid versions of a
//          function an alias is created. `f(long, char, signed char)` would be
//          mangled to
//          `_Z1flca`. The remangler would rename this function to `_Z1fyaa`
//          (`f(long long, signed char, signed char)`). If the target uses a
//          signed char then a valid alias `_Z1fyca`,
//          (`f(long long, char, signed char)`), is not defined. The remangler
//          creates an alias of the renamed function,`_Z1fyaa` , to this
//          permutation, `_Z1fyca`.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <memory>
#include <system_error>

using namespace clang;
using namespace llvm;
using namespace llvm::itanium_demangle;

enum class Signedness { Signed, Unsigned };
enum class SupportedLongWidth { L32, L64 };

static ExitOnError ExitOnErr;

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input bitcode>"));
static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"));
static cl::opt<SupportedLongWidth>
    LongWidth("long-width",
              cl::values(clEnumValN(SupportedLongWidth::L32, "l32",
                                    "long is 32-bit wide."),
                         clEnumValN(SupportedLongWidth::L64, "l64",
                                    "long is 64-bit wide.")));
static cl::opt<Signedness> CharSignedness(
    "char-signedness",
    cl::values(clEnumValN(Signedness::Signed, "signed", "char is signed."),
               clEnumValN(Signedness::Unsigned, "unsigned",
                          "char is unsigned.")));
static cl::opt<bool> Verbose("v", cl::desc("Enable verbose output"),
                             cl::init(false));

namespace {
class BumpPointerAllocator {
  struct BlockMeta {
    BlockMeta *Next;
    size_t Current;
  };

  static constexpr size_t AllocSize = 4096;
  static constexpr size_t UsableAllocSize = AllocSize - sizeof(BlockMeta);

  alignas(long double) char InitialBuffer[AllocSize];
  BlockMeta *BlockList = nullptr;

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

public:
  BumpPointerAllocator()
      : BlockList(new (InitialBuffer) BlockMeta{nullptr, 0}) {}

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
};

class DefaultAllocator {
  BumpPointerAllocator Alloc;

public:
  void reset() { Alloc.reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&... args) {
    return new (Alloc.allocate(sizeof(T))) T(std::forward<Args>(args)...);
  }

  void *allocateNodeArray(size_t sz) {
    return Alloc.allocate(sizeof(Node *) * sz);
  }
};
} // unnamed namespace

using Demangler = ManglingParser<DefaultAllocator>;

class Remangler {
  const Node *Root;
  SmallDenseMap<const char *, const char *> TypeReplacements;

  SmallVector<std::string, 16> Subs;
  bool Failed = false;

  OutputBuffer printNode(const Node *node) {
    OutputBuffer nodeOutBuffer;
    initializeOutputBuffer(nullptr, nullptr, nodeOutBuffer, 1024);
    node->print(nodeOutBuffer);
    return nodeOutBuffer;
  }

  void addSub(const Node *node) {
    OutputBuffer nodeOut = printNode(node);
    char *nodeOutBuf = nodeOut.getBuffer();
    auto nodeOutStr =
        std::string(nodeOutBuf, nodeOutBuf + nodeOut.getCurrentPosition());
    free(nodeOutBuf);
    Subs.push_back(nodeOutStr);
  }

  bool findSub(const Node *node, size_t *index) {
    OutputBuffer nodeOut = printNode(node);
    char *nodeOutBuf = nodeOut.getBuffer();
    auto nodeOutStr =
        std::string(nodeOutBuf, nodeOutBuf + nodeOut.getCurrentPosition());
    free(nodeOutBuf);

    for (size_t i = Subs.size(); i > 0; --i) {
      if (nodeOutStr == Subs[i - 1]) {
        *index = i - 1;
        return true;
      }
    }
    return false;
  }

  bool remangleSub(const Node *node, OutputBuffer &OB) {
    size_t index = 0;
    if (findSub(node, &index)) {
      OB << 'S';
      if (index != 0)
        OB << index;
      OB << '_';
      return true;
    }
    return false;
  }

  void remangleOpenCLCName(const Node *nameNode, OutputBuffer &OB,
                           bool Substitutable, bool isNameRoot = true) {
    if (Substitutable && remangleSub(nameNode, OB))
      return;
    switch (nameNode->getKind()) {
    case Node::Kind::KNameType: {
      const NameType *name = static_cast<const NameType *>(nameNode);
      OB << name->getName().size();
      OB << name->getName();
      break;
    }
    case Node::Kind::KNestedName: {
      if (isNameRoot)
        OB << 'N';
      const NestedName *nestedName = static_cast<const NestedName *>(nameNode);
      remangleOpenCLCName(nestedName->Qual, OB, Substitutable,
                          /* isNameRoot= */ false);
      remangleOpenCLCName(nestedName->Name, OB, /* Substitutable= */ false,
                          /* isNameRoot= */ false);
      if (isNameRoot)
        OB << 'E';
      break;
    }
    case Node::Kind::KNameWithTemplateArgs: {
      const NameWithTemplateArgs *templateName =
          static_cast<const NameWithTemplateArgs *>(nameNode);
      assert(templateName->TemplateArgs->getKind() ==
             Node::Kind::KTemplateArgs);
      remangleOpenCLCName(templateName->Name, OB, /* Substitutable= */ false,
                          /* isNameRoot= */ false);
      OB << 'I';
      const TemplateArgs *templateArgs =
          static_cast<const TemplateArgs *>(templateName->TemplateArgs);
      for (auto templateArgType : templateArgs->getParams())
        remangleOpenCLCType(templateArgType, OB);
      OB << 'E';
      break;
    }
    default: {
      OutputBuffer errorTypeOut;
      initializeOutputBuffer(nullptr, nullptr, errorTypeOut, 1024);
      errorTypeOut << "Unhandled name : ";
      nameNode->print(errorTypeOut);
      errorTypeOut << "\n";
      errs() << errorTypeOut.getBuffer();
      free(errorTypeOut.getBuffer());
      Failed = true;
    }
    }
    if (Substitutable)
      addSub(nameNode);
  }

  void remangleOpenCLCTypeName(const NameType *typeName, OutputBuffer &OB) {
    StringView name = typeName->getName();

    auto it = TypeReplacements.find(name.begin());
    if (it != TypeReplacements.end())
      name = StringView(it->second);

    if (name == "void")
      OB << 'v';
    else if (name == "wchar_t")
      OB << 'w';
    else if (name == "bool")
      OB << 'b';
    else if (name == "char")
      OB << 'c';
    else if (name == "signed char")
      OB << 'a';
    else if (name == "unsigned char")
      OB << 'h';
    else if (name == "short")
      OB << 's';
    else if (name == "unsigned short")
      OB << 't';
    else if (name == "int")
      OB << 'i';
    else if (name == "unsigned int")
      OB << 'j';
    else if (name == "long")
      OB << 'l';
    else if (name == "unsigned long")
      OB << 'm';
    else if (name == "long long")
      OB << 'x';
    else if (name == "unsigned long long")
      OB << 'y';
    else if (name == "__int128")
      OB << 'n';
    else if (name == "unsigned __int128")
      OB << 'o';
    else if (name == "float")
      OB << 'f';
    else if (name == "double")
      OB << 'd';
    else if (name == "long double")
      OB << 'e';
    else if (name == "__float128")
      OB << 'g';
    else if (name == "...")
      OB << 'z';
    // TODO: u
    else if (name == "decimal64")
      OB << "Dd";
    else if (name == "decimal128")
      OB << "De";
    else if (name == "decimal32")
      OB << "Df";
    else if (name == "decimal16")
      OB << "Dh";
    else if (name == "char32_t")
      OB << "Di";
    else if (name == "char16_t")
      OB << "Ds";
    else if (name == "char8_t")
      OB << "Du";
    else if (name == "_Float16")
      OB << "DF16_";
    else if (name == "auto")
      OB << 'a';
    else if (name == "decltype(auto)")
      OB << 'c';
    else if (name == "std::nullptr_t")
      OB << 'n';
    // Enum
    else
      remangleOpenCLCName(typeName, OB, /* Substitutable= */ true);
  }

  void remangleOpenCLCQualifiers(const itanium_demangle::Qualifiers quals,
                                 OutputBuffer &OB) {
    if (quals & QualConst)
      OB << "K";
    if (quals & QualVolatile)
      OB << "V";
    if (quals & QualRestrict)
      OB << "r";
  }

  void remangleOpenCLCType(const Node *typeNode, OutputBuffer &OB) {
    switch (typeNode->getKind()) {
    case Node::Kind::KPointerType: {
      const itanium_demangle::PointerType *ptype =
          static_cast<const itanium_demangle::PointerType *>(typeNode);
      OB << 'P';
      remangleOpenCLCType(ptype->getPointee(), OB);
      break;
    }
    case Node::Kind::KVectorType: {
      if (remangleSub(typeNode, OB))
        return;

      const itanium_demangle::VectorType *vecType =
          static_cast<const itanium_demangle::VectorType *>(typeNode);
      assert(vecType->getDimension()->getKind() == Node::Kind::KNameType);
      const NameType *dims =
          static_cast<const NameType *>(vecType->getDimension());
      OB << "Dv";
      OB << dims->getName();
      OB << '_';
      remangleOpenCLCType(vecType->getBaseType(), OB);
      addSub(typeNode);
      break;
    }
    case Node::Kind::KBinaryFPType: {
      if (remangleSub(typeNode, OB))
        return;

      const BinaryFPType *BFPType = static_cast<const BinaryFPType *>(typeNode);
      assert(BFPType->getDimension()->getKind() == Node::Kind::KNameType);
      const NameType *dims =
          static_cast<const NameType *>(BFPType->getDimension());

      OB << "DF";
      OB << dims->getName();
      OB << '_';
      break;
    }
    case Node::Kind::KVendorExtQualType: {
      if (remangleSub(typeNode, OB))
        return;

      const VendorExtQualType *extQualType =
          static_cast<const VendorExtQualType *>(typeNode);
      OB << 'U';
      OB << extQualType->getExt().size();
      OB << extQualType->getExt();
      remangleOpenCLCType(extQualType->getTy(), OB);
      addSub(typeNode);
      break;
    }
    case Node::Kind::KQualType: {
      const itanium_demangle::QualType *qtype =
          static_cast<const itanium_demangle::QualType *>(typeNode);
      remangleOpenCLCQualifiers(qtype->getQuals(), OB);
      remangleOpenCLCType(qtype->getChild(), OB);
      break;
    }
    case Node::Kind::KNameType: {
      const NameType *typeName = static_cast<const NameType *>(typeNode);
      remangleOpenCLCTypeName(typeName, OB);
      break;
    }
    case Node::Kind::KNestedName: {
      // Enum type with nested name
      remangleOpenCLCName(typeNode, OB, /* Substitutable= */ true);
      break;
    }
    default: {
      OutputBuffer errorTypeOut;
      initializeOutputBuffer(nullptr, nullptr, errorTypeOut, 1024);
      errorTypeOut << "Unhandled type : ";
      typeNode->print(errorTypeOut);
      errorTypeOut << "\n";
      errs() << errorTypeOut.getBuffer();
      free(errorTypeOut.getBuffer());
      Failed = true;
    }
    }
  }

  void remangleOpenCLCFunction(const Node *root, OutputBuffer &OB) {
    assert(root->getKind() == Node::Kind::KFunctionEncoding);
    OB << "_Z";

    const FunctionEncoding *encoding =
        static_cast<const FunctionEncoding *>(root);

    remangleOpenCLCName(encoding->getName(), OB, /* Substitutable= */ false);

    if (encoding->getReturnType())
      remangleOpenCLCType(encoding->getReturnType(), OB);

    for (const Node *paramType : encoding->getParams())
      remangleOpenCLCType(paramType, OB);

    if (encoding->getParams().size() == 0)
      OB << 'v';
  }

public:
  Remangler(const Node *root,
            SmallDenseMap<const char *, const char *> typeReplacements)
      : Root(root), TypeReplacements(typeReplacements) {}

  bool hasFailed() { return Failed; }

  std::string remangle() {
    Subs.clear();
    OutputBuffer remanglingStream;
    initializeOutputBuffer(nullptr, nullptr, remanglingStream, 1024);
    remangleOpenCLCFunction(Root, remanglingStream);
    std::string remangled = std::string(remanglingStream.getBuffer(),
                                        remanglingStream.getCurrentPosition());
    free(remanglingStream.getBuffer());
    return remangled;
  }
};

class TargetTypeReplacements {
  SmallDenseMap<const char *, const char *> ParameterTypeReplacements;
  SmallDenseMap<const char *, const char *> AliasTypeReplacements;
  SmallDenseMap<const char *, const char *> RemangledAliasTypeReplacements;

  void CreateRemangledTypeReplacements() {
    // RemangleTypes which are not aliases or not the exact same alias type
    for (auto &TypeReplacementPair : ParameterTypeReplacements)
      if (AliasTypeReplacements.find(TypeReplacementPair.getFirst()) ==
          AliasTypeReplacements.end())
        RemangledAliasTypeReplacements[TypeReplacementPair.getFirst()] =
            TypeReplacementPair.getSecond();
      else if (AliasTypeReplacements[TypeReplacementPair.getFirst()] !=
               TypeReplacementPair.getSecond())
        RemangledAliasTypeReplacements[TypeReplacementPair.getFirst()] =
            TypeReplacementPair.getSecond();
  }

public:
  TargetTypeReplacements() {
    // Replace long with long long
    ParameterTypeReplacements["long"] = "long long";
    ParameterTypeReplacements["unsigned long"] = "unsigned long long";

    // Replace char with signed char
    ParameterTypeReplacements["char"] = "signed char";

    // Make replaced long functions aliases to either integer or long long
    // variant
    if (LongWidth == SupportedLongWidth::L32) {
      AliasTypeReplacements["long"] = "int";
      AliasTypeReplacements["unsigned long"] = "unsigned int";
    } else {
      AliasTypeReplacements["long"] = "long long";
      AliasTypeReplacements["unsigned long"] = "unsigned long long";
    }

    // Make replaced char functions aliases to either integer or long long
    // variant
    if (CharSignedness == Signedness::Signed) {
      AliasTypeReplacements["char"] = "signed char";
    } else {
      AliasTypeReplacements["char"] = "unsigned char";
    }

    CreateRemangledTypeReplacements();
  }

  SmallDenseMap<const char *, const char *> getParameterTypeReplacements() {
    return ParameterTypeReplacements;
  }

  SmallDenseMap<const char *, const char *> getAliasTypeReplacements() {
    return AliasTypeReplacements;
  }

  SmallDenseMap<const char *, const char *>
  getRemangledAliasTypeReplacements() {
    return RemangledAliasTypeReplacements;
  }
};

bool createAliasFromMap(
    Module *M, std::string originalName,
    const itanium_demangle::Node *functionTree,
    SmallDenseMap<const char *, const char *> TypeReplacements,
    bool AliaseeTypeReplacement = false) {
  Remangler ATR{functionTree, TypeReplacements};
  std::string RemangledName = ATR.remangle();

  if (ATR.hasFailed())
    return false;

  // Name has not changed from the original name.
  if (RemangledName == originalName)
    return true;

  StringRef AliasName, AliaseeName;
  if (AliaseeTypeReplacement) {
    AliasName = originalName;
    AliaseeName = RemangledName;
  } else {
    AliasName = RemangledName;
    AliaseeName = originalName;
  }

  Function *Aliasee = M->getFunction(AliaseeName);
  if (Aliasee) {
    GlobalAlias::create(AliasName, Aliasee);
  } else if (Verbose) {
    std::cout << "Could not create alias " << AliasName.data() << " : missing "
              << AliaseeName.data() << std::endl;
  }

  return true;
}

bool createAliases(Module *M, std::string originalMangledName,
                   std::string remangledName,
                   const itanium_demangle::Node *functionTree,
                   TargetTypeReplacements replacements) {
  // create alias of original function
  if (!createAliasFromMap(M, originalMangledName, functionTree,
                          replacements.getAliasTypeReplacements(),
                          /* AliaseeTypeReplacement= */ true))
    return false;

  // create alias from remangled function
  if (!createAliasFromMap(M, remangledName, functionTree,
                          replacements.getRemangledAliasTypeReplacements()))
    return false;

  return true;
}

bool remangleFunction(Function &func, Module *M,
                      TargetTypeReplacements replacements) {
  if (!func.getName().startswith("_Z"))
    return true;

  std::string MangledName = func.getName().str();
  Demangler D{MangledName.c_str(), MangledName.c_str() + MangledName.length()};
  const itanium_demangle::Node *FunctionTree = D.parse();
  if (FunctionTree == nullptr) {
    errs() << "Unable to demangle name: " << MangledName << '\n';
    return false;
  }

  // Try to change the parameter types in the function name using the mappings.
  Remangler R{FunctionTree, replacements.getParameterTypeReplacements()};
  std::string RemangledName = R.remangle();

  if (R.hasFailed())
    return false;

  if (RemangledName != MangledName) {
    if (Verbose) {
      std::cout << "Mangling changed:"
                << "\n"
                << "Original:  " << MangledName << "\n"
                << "New:       " << RemangledName << "\n"
                << std::endl;
    }
    func.setName(RemangledName);

    // Make an alias to a suitable function using the old name if there is a
    // type-mapping and the corresponding aliasee function exists.
    if (!createAliases(M, MangledName, RemangledName, FunctionTree,
                       replacements))
      return false;
  }

  return true;
}

int main(int argc, const char **argv) {
  LLVMContext Context;

  ExitOnErr.setBanner(std::string(argv[0]) + ": error: ");
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::ParseCommandLineOptions(
      argc, argv, "Tool for remangling libclc bitcode to fit a target\n");

  if (OutputFilename.empty()) {
    errs() << "No output file.\n";
    return 1;
  }

  TargetTypeReplacements Replacements;

  std::string ErrorMessage;
  std::unique_ptr<MemoryBuffer> BufferPtr =
      ExitOnErr(errorOrToExpected(MemoryBuffer::getFileOrSTDIN(InputFilename)));
  std::unique_ptr<llvm::Module> M =
      ExitOnErr(parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context));

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out(
      new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  bool Success = true;
  for (auto &Func : M->getFunctionList())
    Success = remangleFunction(Func, M.get(), Replacements) && Success;

  // Only fail after all to give as much context as possible.
  if (!Success) {
    errs() << "Failed to remangle all mangled functions in module.\n";
    return 1;
  }

  WriteBitcodeToFile(*M, Out->os());

  // Declare success.
  Out->keep();
  return 0;
}
