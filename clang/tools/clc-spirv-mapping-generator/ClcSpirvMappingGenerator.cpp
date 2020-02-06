//=== ClcSpirvMappingGenerator.cpp - CLC to SPIR-V Mapping Generator Tool -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// This file implements a utility for generating a `mapping.json` containing
/// the information required to produce interfaces that translate between
/// OpenCL built-ins and SPIR-V built-ins.
///
/// It does this by generating a LLVM module containing wrapper functions that
/// call the OpenCL built-ins defined in the opencl-c.h header. This wrapper
/// module is then provided to the LLVM IR to SPIR-V translator to produce a
/// module with wrapper functions where the definitions have been replaced with
/// calls to `__spirv` functions. The module is then used to extract a mapping
/// between the wrapper functions (which have maintained their OpenCL built-in
/// naming) and the modified definitions which contain `__spirv` calls, which is
/// written to a JSON file.
///
/// You can run this with:
/// $BUILD_DIR/bin/clc-spirv-mapping-generator $BUILD_DIR/lib/clang/9.0.0/include/opencl-c.h

#include "LLVMSPIRVLib.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_os_ostream.h"
#include <sstream>

using namespace clang;
using namespace clang::driver;
using namespace clang::CodeGen;
using namespace clang::tooling;
using namespace llvm;

static ExitOnError ExitOnErr("An error has occured.\n");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

static cl::OptionCategory
    ClcSpirvMappingCategory("clc-spirv-mapping-generator options");

// Helper function to produce the mangled name of a `NamedDecl` with and without
// `wrapper_` being prefixed.
std::tuple<std::string, std::string> GetMangledName(MangleContext &MangleCtx,
                                                    IdentifierTable *IdentTable,
                                                    NamedDecl *D) {
  SmallString<256> Buffer;
  std::string MangledName;
  std::string MangledNameWithWrapper;
  if (MangleCtx.shouldMangleDeclName(D)) {
    llvm::raw_svector_ostream Out(Buffer);

    MangleCtx.mangleName(D, Out);
    MangledName = std::string(Out.str());
    Buffer.clear();

    DeclarationName OriginalName = D->getDeclName();
    std::string WrappedNameStr = "wrapper_" + OriginalName.getAsString();
    const IdentifierInfo &II = IdentTable->get(WrappedNameStr);
    DeclarationName WrappedName = DeclarationName(&II);
    D->setDeclName(WrappedName);

    MangleCtx.mangleName(D, Out);
    MangledNameWithWrapper = std::string(Out.str());
    D->setDeclName(OriginalName);
  } else {
    IdentifierInfo *II = D->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");
    MangledName = std::string(II->getName());
    MangledNameWithWrapper = ("wrapper_" + II->getName()).str();
  }

  return std::make_tuple(MangledName, MangledNameWithWrapper);
}

class PopulateModuleAction : public ASTConsumer,
                             public RecursiveASTVisitor<PopulateModuleAction> {
public:
  PopulateModuleAction(llvm::LLVMContext &Ctx, CodeGenerator &CG,
                       llvm::Module &Output)
      : Ctx(Ctx), CG(CG), Output(Output) {
    IdentTable = nullptr;
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  void HandleTranslationUnit(ASTContext &Context) override {
    IdentTable = &Context.Idents;
    MangleCtx.reset(Context.createMangleContext());
    CG.Initialize(Context);

    // Pretend to be OpenCL C++ for the translator.
    llvm::IntegerType *Int32Ty = llvm::Type::getInt32Ty(Ctx);
    llvm::Metadata *SourceElts[] = {
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 4)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 100000))};
    llvm::NamedMDNode *SourceMetadata =
        Output.getOrInsertNamedMetadata("spirv.Source");
    SourceMetadata->addOperand(llvm::MDNode::get(Ctx, SourceElts));

    TraverseDecl(Context.getTranslationUnitDecl());

    // Make sure the output module is also targetting `spirv64`.
    Output.setTargetTriple(CG.GetModule()->getTargetTriple());
    Output.setDataLayout(CG.GetModule()->getDataLayout());

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    // Confirm that we've constructed a valid module.
    SmallString<256> Buffer;
    llvm::raw_svector_ostream Out(Buffer);
    if (verifyModule(Output, &Out)) {
      errs() << "Module:\n";
      Output.dump();
      errs() << "\n";
      std::string VerificationError = std::string(Out.str());
      ExitOnErr(llvm::make_error<StringError>(
          VerificationError, llvm::make_error_code(llvm::errc::io_error)));
    }
#endif
  }

  bool VisitNamedDecl(NamedDecl *D) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
    if (!FD) {
      return true;
    }

    // Generate the LLVM IR for the function.
    llvm::Constant *ActualFunctionConst =
        CG.GetAddrOfGlobal(GlobalDecl(FD), true);
    Function *ActualFunction = dyn_cast<Function>(ActualFunctionConst);
    if (!ActualFunction) {
      return true;
    }

    // Skip functions that are problematic for the translator.
    if (FunctionsToSkip.find(D->getDeclName().getAsString()) !=
        FunctionsToSkip.end()) {
      return true;
    }

    // Generate the mangled name of the function with and without the wrapper
    // prefix.
    SmallString<256> Buffer;
    std::string MangledName, MangledNameWithWrapper;
    std::tie(MangledName, MangledNameWithWrapper) =
        GetMangledName(*MangleCtx, IdentTable, D);

    // Check if a function with the same name already exists, this can happen
    // for some overloads that the mangling doesn't disambiguate between.
    //
    // See `clamp` function in output w/out this check.
    if (Output.getFunction(MangledName) != nullptr) {
      return true;
    }

    // Link to this function in the output module.
    Function *ActualFunctionInOutput =
        Function::Create(ActualFunction->getFunctionType(),
                         GlobalValue::ExternalLinkage, MangledName, Output);
    ActualFunctionInOutput->setAttributes(ActualFunction->getAttributes());

    // Insert the wrapper function.
    FunctionCallee WrappedFunctionCallee = Output.getOrInsertFunction(
        MangledNameWithWrapper, ActualFunctionInOutput->getFunctionType(),
        ActualFunctionInOutput->getAttributes());

    Function *WrappedFunction =
        dyn_cast<Function>(WrappedFunctionCallee.getCallee());
    if (!WrappedFunction) {
      return true;
    }

    // Create a body for the wrapper function.
    BasicBlock *BB =
        BasicBlock::Create(Ctx, Twine{"entry"}, WrappedFunction, nullptr);

    std::vector<Value *> Args;
    for (Argument &A : WrappedFunction->args())
      Args.push_back(&A);

    // Insert a call to the original function.
    CallInst *CI = CallInst::Create(ActualFunctionInOutput->getFunctionType(),
                                    ActualFunctionInOutput, Args, "", BB);

    // Return the result of the call if the function doesn't return void.
    ReturnInst::Create(
        Ctx, WrappedFunction->getReturnType()->isVoidTy() ? nullptr : CI, BB);

    return true;
  }

  typedef std::set<std::string> FuncSet;
  static FuncSet FunctionsToSkip;

private:
  llvm::LLVMContext &Ctx;
  CodeGenerator &CG;
  llvm::Module &Output;

  IdentifierTable *IdentTable;
  std::unique_ptr<MangleContext> MangleCtx;
};

// TODO(davidtwco): Find a way to support these functions.
PopulateModuleAction::FuncSet PopulateModuleAction::FunctionsToSkip = {
    // The translator requires that the argument to these functions be a
    // constant int that it
    // can read the value of and not an argument (which would be the case w/ a
    // wrapper function).
    "barrier",
    "mem_fence",
    "read_mem_fence",
    "write_mem_fence",
    "sub_group_barrier",
    "work_group_barrier",
    // The translator tries to convert these function to the same name as the
    // SPIR-V built-in for
    // another function but with a different signature which results in an
    // attempt to re-define a
    // function.
    "read_imagei",  // conflicts with `read_imagef`
    "read_imageui", // conflicts with `read_imagef`
    "read_imageh",  // conflicts with `read_imagef`
    // The translator doesn't know how to demangle these names correctly.
    "amd_bfe",
};

class PopulateModuleActionFactory {
public:
  PopulateModuleActionFactory(llvm::LLVMContext &Ctx, CodeGenerator &CG,
                              llvm::Module &Output)
      : Ctx(Ctx), CG(CG), Output(Output) {}

  std::unique_ptr<clang::ASTConsumer> newASTConsumer() {
    return std::make_unique<PopulateModuleAction>(Ctx, CG, Output);
  }

private:
  llvm::LLVMContext &Ctx;
  CodeGenerator &CG;
  llvm::Module &Output;
};

// Helper function to create a `ClangTool` with the correct arguments.
ClangTool getTool(CommonOptionsParser &OptionsParser) {
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  Tool.appendArgumentsAdjuster(getClangStripOutputAdjuster());
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-xcl", ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-cl-std=CL1.2"));
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-fno-builtin"));
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-Dcl_khr_int64_base_atomics"));
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-Dcl_khr_int64_extended_atomics"));
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-Dcl_khr_3d_image_writes"));
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-Dcl_khr_fp16"));
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-target"));
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("spir64-unknown-unknown"));

  return Tool;
}

// Helper function to create a `CodeGenerator` for generating LLVM IR from the
// `opencl-c.h`.
std::unique_ptr<CodeGenerator> getCodeGenerator(llvm::LLVMContext &Ctx) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  auto DiagEngine = std::make_unique<DiagnosticsEngine>(DiagIDs, DiagOpts);
  auto HSO = std::make_unique<HeaderSearchOptions>();
  auto PO = std::make_unique<PreprocessorOptions>();
  auto CGO = std::make_unique<CodeGenOptions>();

  auto CG = std::unique_ptr<CodeGenerator>(CreateLLVMCodeGen(
      *DiagEngine, StringRef{"opencl-c.h"}, *HSO, *PO, *CGO, Ctx, nullptr));
  return CG;
}

// Populate `Output` with a wrapper function for the file specified in
// `OptionsParser`.
int populateModule(CommonOptionsParser &OptionsParser, llvm::LLVMContext &Ctx,
                   llvm::Module &Output) {
  auto Tool = getTool(OptionsParser);
  auto CG = getCodeGenerator(Ctx);

  PopulateModuleActionFactory PopulateModuleFactory(Ctx, *CG, Output);
  std::unique_ptr<FrontendActionFactory> FrontendFactory;
  FrontendFactory = newFrontendActionFactory(&PopulateModuleFactory);

  return Tool.run(FrontendFactory.get());
}

// Buffer that will ignore the output of llvm-spirv buffer
template <typename Ch, typename Traits = std::char_traits<Ch>>
struct basic_nullbuf : std::basic_streambuf<Ch, Traits> {
  typedef std::basic_streambuf<Ch, Traits> base_type;
  typedef typename base_type::int_type int_type;
  typedef typename base_type::traits_type traits_type;

  virtual int_type overflow(int_type c) { return traits_type::not_eof(c); }
};

typedef basic_nullbuf<char> nullbuf;

// Run the module through the translator, ignoring the produced SPIR-V and
// then using the mangled LLVM IR that it leaves behind.
void translateModule(llvm::Module &WrapperModule) {
  nullbuf nullObject;
  std::ostream OS(&nullObject);

  auto TranslateToErr = std::string();
  if (!llvm::writeSpirv(&WrapperModule, OS, TranslateToErr)) {
    ExitOnErr(llvm::make_error<StringError>(
        TranslateToErr, llvm::make_error_code(llvm::errc::io_error)));
  }
}

struct ClcSpirvArgument {
  // Type of the argument.
  std::string Type;
  // If the argument is a constant, then this contains the value of the
  // constant.
  llvm::Optional<std::string> IsConst;
};

struct ClcSpirvMapping {
  // Mangled name of the OpenCL built-in.
  std::string MangledName;
  // Unmangled name of the OpenCL built-in.
  std::string UnmangledName;
  // Mangled name of the SPIR-V built-in, if it is used in a call instruction or
  // exists at all.
  llvm::Optional<std::string> MangledSpirvName;
  // Vector with the type of the arguments to the OpenCL function.
  std::vector<std::string> OpenClArguments;
  // Return type of the OpenCL function.
  std::string OpenClReturnType;
  // Vector of the information from the SPIR-V arguments. This will be empty if
  // there was no call to a `__spirv_*` Function
  std::vector<ClcSpirvArgument> SpirvArguments;
  // Return type of the SPIR-V function.
  llvm::Optional<std::string> SpirvReturnType;
  // String containing the LLVM IR of the function body.
  std::string Ir;
};

using ClcSpirvMappingMap = llvm::StringMap<ClcSpirvMapping>;

class CollectModuleAction : public ASTConsumer,
                            public RecursiveASTVisitor<CollectModuleAction> {
public:
  CollectModuleAction(llvm::Module &Input, ClcSpirvMappingMap &Mapping)
      : Input(Input), Mapping(Mapping) {
    IdentTable = nullptr;
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  void HandleTranslationUnit(ASTContext &Context) override {
    IdentTable = &Context.Idents;
    MangleCtx.reset(Context.createMangleContext());

    TraverseDecl(Context.getTranslationUnitDecl());
  }

  bool VisitNamedDecl(NamedDecl *D) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
    if (!FD) {
      return true;
    }

    // Generate the mangled name of the function with and without the wrapper
    // prefix.
    auto UnmangledName = D->getDeclName().getAsString();
    std::string MangledName, MangledNameWithWrapper;
    std::tie(MangledName, MangledNameWithWrapper) =
        GetMangledName(*MangleCtx, IdentTable, D);

    // Skip functions that we didn't generate input for.
    Function *F = Input.getFunction(MangledNameWithWrapper);
    if (F == nullptr) {
      return true;
    }

    // Skip functions that we already have an entry for.
    if (Mapping.count(MangledName)) {
      return true;
    }

    std::pair<ClcSpirvMappingMap::iterator, bool> iter =
        Mapping.try_emplace(MangledName);
    auto &FunctionMapping = iter.first->second;
    if (iter.second) {
      FunctionMapping.MangledName = MangledName;
      FunctionMapping.UnmangledName = UnmangledName;

      for (Argument &A : F->args()) {
        FunctionMapping.OpenClArguments.emplace_back(
            GetTypeAsString(A.getType()));
      }

      FunctionMapping.OpenClReturnType = GetTypeAsString(F->getReturnType());

      for (BasicBlock &BB : *F) {
        FunctionMapping.Ir = GetValueAsString(&BB);

        for (Instruction &I : BB) {
          CallInst *CI = dyn_cast<CallInst>(&I);
          if (!CI) {
            continue;
          }

          auto CalledFunctionName = CI->getCalledFunction()->getName();
          auto IsSpirvCall =
              CalledFunctionName.find("__spirv_") != std::string::npos;
          if (!IsSpirvCall) {
            continue;
          }

          FunctionMapping.MangledSpirvName = std::string(CalledFunctionName);
          FunctionMapping.SpirvReturnType =
              GetTypeAsString(CI->getCalledFunction()->getReturnType());
          FunctionMapping.SpirvArguments.clear();

          for (Value *Operand : CI->arg_operands()) {
            Constant *C = dyn_cast<Constant>(Operand);
            auto IsConst = llvm::Optional<std::string>();
            if (C) {
              IsConst = llvm::Optional<std::string>(GetValueAsString(C, false));
            }

            auto Type = GetTypeAsString(Operand->getType());

            ClcSpirvArgument Arg = {Type, IsConst};
            FunctionMapping.SpirvArguments.emplace_back(Arg);
          }
        }
      }
    }

    return true;
  }

private:
  llvm::Module &Input;
  ClcSpirvMappingMap &Mapping;

  IdentifierTable *IdentTable;
  std::unique_ptr<MangleContext> MangleCtx;

  // Helper function to print a value to a string.
  std::string GetValueAsString(Value *V) {
    SmallString<256> IrBuffer;
    llvm::raw_svector_ostream Out(IrBuffer);
    V->print(Out);
    return NormalizeNewlines(std::string(Out.str()));
  }

  // Helper function to print a value to a string, with or without the type.
  std::string GetValueAsString(Value *V, bool WithType) {
    SmallString<256> ValueBuffer;
    llvm::raw_svector_ostream Out(ValueBuffer);
    V->printAsOperand(Out, WithType, &Input);
    return NormalizeNewlines(std::string(Out.str()));
  }

  // Helper function to print a type to a string.
  std::string GetTypeAsString(llvm::Type *T) {
    SmallString<256> TypeBuffer;
    llvm::raw_svector_ostream Out(TypeBuffer);
    T->print(Out);
    return NormalizeNewlines(std::string(Out.str()));
  }

  // Helper function to convert a multi-line string into a single line string
  // with escaped newlines.
  std::string NormalizeNewlines(std::string Input) {
    std::string::size_type pos = 0;
    while ((pos = Input.find("\n", pos)) != std::string::npos) {
      Input.replace(pos, 1, "\\n");
    }
    return Input;
  }
};

class CollectModuleActionFactory {
public:
  CollectModuleActionFactory(llvm::Module &Input, ClcSpirvMappingMap &Mapping)
      : Input(Input), Mapping(Mapping) {}

  std::unique_ptr<clang::ASTConsumer> newASTConsumer() {
    return std::make_unique<CollectModuleAction>(Input, Mapping);
  }

private:
  llvm::Module &Input;
  ClcSpirvMappingMap &Mapping;
};

// Collect the mapping from the wrapper module into a `ClcSpirvMappingMap`.
int collectModule(CommonOptionsParser &OptionsParser, llvm::Module &Input,
                  ClcSpirvMappingMap &Mapping) {
  auto Tool = getTool(OptionsParser);

  CollectModuleActionFactory CollectModuleFactory(Input, Mapping);
  std::unique_ptr<FrontendActionFactory> FrontendFactory;
  FrontendFactory = newFrontendActionFactory(&CollectModuleFactory);

  return Tool.run(FrontendFactory.get());
}

class MappingPrinter {
public:
  MappingPrinter(const ClcSpirvMappingMap &Mapping, raw_ostream &S)
      : Mapping(Mapping), Stream(S) {}

  struct JSONEntry {
    JSONEntry(raw_ostream &Stream) : Stream(Stream), addedValues(0) {}

    inline void emitSeparator() {
      if (addedValues)
        Stream << ", ";
      addedValues++;
    }

  protected:
    raw_ostream &Stream;
    unsigned addedValues;
  };

  struct CreateMap : public JSONEntry {
    CreateMap(raw_ostream &Stream) : JSONEntry(Stream) { Stream << "{"; }

    inline void add(const Twine key, const Twine value) {
      emitSeparator();
      Stream << "\"" << key << "\": \"" << value << "\"";
    }

    inline void add(const Twine key, const bool value) {
      emitSeparator();
      Stream << "\"" << key << "\": " << (value ? "true" : "false");
    }

    inline void add(const StringRef key, std::function<void()> SubEntry) {
      emitSeparator();
      Stream << "\"" << key << "\": ";
      SubEntry();
    }

    ~CreateMap() { Stream << "}"; }
  };

  struct CreateList : public JSONEntry {
    CreateList(raw_ostream &Stream) : JSONEntry(Stream) { Stream << "["; }

    inline void add(const bool value) {
      emitSeparator();
      Stream << (value ? "true" : "false");
    }

    inline void add(const Twine value) {
      emitSeparator();
      Stream << "\"" << value << "\"";
    }

    inline void add(std::function<void()> SubEntry) {
      emitSeparator();
      SubEntry();
    }

    ~CreateList() { Stream << "]"; }
  };

  void run() {
    CreateMap Document(Stream);
    for (const StringMapEntry<ClcSpirvMapping> &WrappedEntry : Mapping) {
      const ClcSpirvMapping &Entry = WrappedEntry.getValue();

      Document.add(Entry.MangledName, [&]() {
        CreateMap Map(Stream);

        Map.add("MangledName", Entry.MangledName);
        Map.add("UnmangledName", Entry.UnmangledName);
        Map.add("OpenClReturnType", Entry.OpenClReturnType);
        Map.add("OpenClArguments", [&]() {
          CreateList List(Stream);
          for (unsigned i = 0; i < Entry.OpenClArguments.size(); ++i) {
            List.add([&]() {
              CreateMap ArgMap(Stream);
              ArgMap.add("Type", Entry.OpenClArguments[i]);
            });
          }
        });
        if (Entry.MangledSpirvName) {
          Map.add("MangledSpirvName", Entry.MangledSpirvName.getValue());
          Map.add("SpirvReturnType", Entry.SpirvReturnType.getValue());
          Map.add("SpirvArguments", [&]() {
            CreateList List(Stream);
            for (unsigned i = 0; i < Entry.SpirvArguments.size(); ++i) {
              List.add([&]() {
                CreateMap ArgMap(Stream);
                ArgMap.add("Type", Entry.SpirvArguments[i].Type);
                ArgMap.add("IsConstant",
                           Entry.SpirvArguments[i].IsConst.hasValue());
                if (Entry.SpirvArguments[i].IsConst) {
                  ArgMap.add("Value",
                             Entry.SpirvArguments[i].IsConst.getValue());
                }
              });
            }
          });
        }
        Map.add("Ir", Entry.Ir);
      });
    }
  }

private:
  const ClcSpirvMappingMap &Mapping;
  raw_ostream &Stream;
};

// Write the mapping out as JSON using the `MappingPrinter`.
void writeMappingAsJson(ClcSpirvMappingMap &Mapping) {
  std::error_code EC;
  ToolOutputFile OutFile("mapping.json", EC, sys::fs::F_None);
  ExitOnErr(errorCodeToError(EC));

  MappingPrinter Printer(Mapping, OutFile.os());
  Printer.run();

  OutFile.keep();
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  CommonOptionsParser OptionsParser(argc, argv, ClcSpirvMappingCategory);

  auto Ctx = std::make_unique<llvm::LLVMContext>();
  auto WrapperModule =
      std::make_unique<llvm::Module>(StringRef{"wrapped-opencl-c.h"}, *Ctx);
  auto Mapping = std::make_unique<ClcSpirvMappingMap>();

  errs() << "Populating module with wrapper functions..\n";
  populateModule(OptionsParser, *Ctx, *WrapperModule);
  errs() << "Translating module to SPIR-V as LLVM IR..\n";
  translateModule(*WrapperModule);
  errs() << "Collecting translations into mapping..\n";
  collectModule(OptionsParser, *WrapperModule, *Mapping);
  errs() << "Writing mapping as JSON..\n";
  writeMappingAsJson(*Mapping);
}
