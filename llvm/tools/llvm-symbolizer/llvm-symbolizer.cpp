//===-- llvm-symbolizer.cpp - Simple addr2line-like symbolizer ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility works much like "addr2line". It is able of transforming
// tuples (module name, module offset) to code locations (function name,
// file, line number, column number). It is targeted for compiler-rt tools
// (especially AddressSanitizer and ThreadSanitizer) that can use it
// to symbolize stack traces in their error reports.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace llvm;
using namespace symbolize;

static cl::opt<bool>
ClUseSymbolTable("use-symbol-table", cl::init(true),
                 cl::desc("Prefer names in symbol table to names "
                          "in debug info"));

static cl::opt<FunctionNameKind> ClPrintFunctions(
    "functions", cl::init(FunctionNameKind::LinkageName),
    cl::desc("Print function name for a given address"), cl::ValueOptional,
    cl::values(clEnumValN(FunctionNameKind::None, "none", "omit function name"),
               clEnumValN(FunctionNameKind::ShortName, "short",
                          "print short function name"),
               clEnumValN(FunctionNameKind::LinkageName, "linkage",
                          "print function linkage name (default)"),
               // Sentinel value for unspecified value.
               clEnumValN(FunctionNameKind::LinkageName, "", "")));
static cl::alias ClPrintFunctionsShort("f", cl::desc("Alias for -functions"),
                                       cl::NotHidden, cl::Grouping,
                                       cl::aliasopt(ClPrintFunctions));

static cl::opt<bool>
    ClUseRelativeAddress("relative-address", cl::init(false),
                         cl::desc("Interpret addresses as relative addresses"),
                         cl::ReallyHidden);

static cl::opt<bool>
    ClPrintInlining("inlining", cl::init(true),
                    cl::desc("Print all inlined frames for a given address"));
static cl::alias
    ClPrintInliningAliasI("i", cl::desc("Alias for -inlining"),
                          cl::NotHidden, cl::aliasopt(ClPrintInlining),
                          cl::Grouping);
static cl::alias
    ClPrintInliningAliasInlines("inlines", cl::desc("Alias for -inlining"),
                                cl::NotHidden, cl::aliasopt(ClPrintInlining));

// -basenames, -s
static cl::opt<bool> ClBasenames("basenames", cl::init(false),
                                 cl::desc("Strip directory names from paths"));
static cl::alias ClBasenamesShort("s", cl::desc("Alias for -basenames"),
                                  cl::NotHidden, cl::aliasopt(ClBasenames));

// -demangle, -C, -no-demangle
static cl::opt<bool>
ClDemangle("demangle", cl::init(true), cl::desc("Demangle function names"));
static cl::alias
ClDemangleShort("C", cl::desc("Alias for -demangle"),
                cl::NotHidden, cl::aliasopt(ClDemangle), cl::Grouping);
static cl::opt<bool>
ClNoDemangle("no-demangle", cl::init(false),
             cl::desc("Don't demangle function names"));

static cl::opt<std::string> ClDefaultArch("default-arch", cl::init(""),
                                          cl::desc("Default architecture "
                                                   "(for multi-arch objects)"));

// -obj, -exe, -e
static cl::opt<std::string>
ClBinaryName("obj", cl::init(""),
             cl::desc("Path to object file to be symbolized (if not provided, "
                      "object file should be specified for each input line)"));
static cl::alias
ClBinaryNameAliasExe("exe", cl::desc("Alias for -obj"),
                     cl::NotHidden, cl::aliasopt(ClBinaryName));
static cl::alias
ClBinaryNameAliasE("e", cl::desc("Alias for -obj"),
                   cl::NotHidden, cl::aliasopt(ClBinaryName));


static cl::opt<std::string>
    ClDwpName("dwp", cl::init(""),
              cl::desc("Path to DWP file to be use for any split CUs"));

static cl::list<std::string>
ClDsymHint("dsym-hint", cl::ZeroOrMore,
           cl::desc("Path to .dSYM bundles to search for debug info for the "
                    "object files"));

// -print-address, -addresses, -a
static cl::opt<bool>
ClPrintAddress("print-address", cl::init(false),
               cl::desc("Show address before line information"));
static cl::alias
ClPrintAddressAliasAddresses("addresses", cl::desc("Alias for -print-address"),
                             cl::NotHidden, cl::aliasopt(ClPrintAddress));
static cl::alias
ClPrintAddressAliasA("a", cl::desc("Alias for -print-address"),
                     cl::NotHidden, cl::aliasopt(ClPrintAddress), cl::Grouping);

// -pretty-print, -p
static cl::opt<bool>
    ClPrettyPrint("pretty-print", cl::init(false),
                  cl::desc("Make the output more human friendly"));
static cl::alias ClPrettyPrintShort("p", cl::desc("Alias for -pretty-print"),
                                    cl::NotHidden,
                                    cl::aliasopt(ClPrettyPrint), cl::Grouping);

static cl::opt<int> ClPrintSourceContextLines(
    "print-source-context-lines", cl::init(0),
    cl::desc("Print N number of source file context"));

static cl::opt<bool> ClVerbose("verbose", cl::init(false),
                               cl::desc("Print verbose line info"));

// -adjust-vma
static cl::opt<unsigned long long>
    ClAdjustVMA("adjust-vma", cl::init(0), cl::value_desc("offset"),
                cl::desc("Add specified offset to object file addresses"));

static cl::list<std::string> ClInputAddresses(cl::Positional,
                                              cl::desc("<input addresses>..."),
                                              cl::ZeroOrMore);

static cl::opt<std::string>
    ClFallbackDebugPath("fallback-debug-path", cl::init(""),
                        cl::desc("Fallback path for debug binaries."));

template<typename T>
static bool error(Expected<T> &ResOrErr) {
  if (ResOrErr)
    return false;
  logAllUnhandledErrors(ResOrErr.takeError(), errs(),
                        "LLVMSymbolizer: error reading file: ");
  return true;
}

static bool parseCommand(StringRef InputString, bool &IsData,
                         std::string &ModuleName, uint64_t &ModuleOffset) {
  const char kDelimiters[] = " \n\r";
  ModuleName = "";
  if (InputString.consume_front("CODE ")) {
    IsData = false;
  } else if (InputString.consume_front("DATA ")) {
    IsData = true;
  } else {
    // If no cmd, assume it's CODE.
    IsData = false;
  }
  const char *pos = InputString.data();
  // Skip delimiters and parse input filename (if needed).
  if (ClBinaryName.empty()) {
    pos += strspn(pos, kDelimiters);
    if (*pos == '"' || *pos == '\'') {
      char quote = *pos;
      pos++;
      const char *end = strchr(pos, quote);
      if (!end)
        return false;
      ModuleName = std::string(pos, end - pos);
      pos = end + 1;
    } else {
      int name_length = strcspn(pos, kDelimiters);
      ModuleName = std::string(pos, name_length);
      pos += name_length;
    }
  } else {
    ModuleName = ClBinaryName;
  }
  // Skip delimiters and parse module offset.
  pos += strspn(pos, kDelimiters);
  int offset_length = strcspn(pos, kDelimiters);
  return !StringRef(pos, offset_length).getAsInteger(0, ModuleOffset);
}

// This routine returns section index for an address.
// Assumption: would work ambiguously for object files which have sections not
// assigned to an address(since the same address could belong to various
// sections).
static uint64_t getModuleSectionIndexForAddress(const std::string &ModuleName,
                                                uint64_t Address) {

  // following ModuleName processing was copied from
  // LLVMSymbolizer::getOrCreateModuleInfo().
  // it needs to be refactored to avoid code duplication.
  std::string BinaryName = ModuleName;
  size_t ColonPos = ModuleName.find_last_of(':');
  // Verify that substring after colon form a valid arch name.
  if (ColonPos != std::string::npos) {
    std::string ArchStr = ModuleName.substr(ColonPos + 1);
    if (Triple(ArchStr).getArch() != Triple::UnknownArch) {
      BinaryName = ModuleName.substr(0, ColonPos);
    }
  }

  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(BinaryName);

  if (error(BinaryOrErr))
    return object::SectionedAddress::UndefSection;

  Binary &Binary = *BinaryOrErr->getBinary();

  if (ObjectFile *O = dyn_cast<ObjectFile>(&Binary)) {
    for (SectionRef Sec : O->sections()) {
      if (!Sec.isText() || Sec.isVirtual())
        continue;

      if (Address >= Sec.getAddress() &&
          Address <= Sec.getAddress() + Sec.getSize()) {
        return Sec.getIndex();
      }
    }
  }

  return object::SectionedAddress::UndefSection;
}

static void symbolizeInput(StringRef InputString, LLVMSymbolizer &Symbolizer,
                           DIPrinter &Printer) {
  bool IsData = false;
  std::string ModuleName;
  uint64_t Offset = 0;
  if (!parseCommand(StringRef(InputString), IsData, ModuleName, Offset)) {
    outs() << InputString;
    return;
  }

  if (ClPrintAddress) {
    outs() << "0x";
    outs().write_hex(Offset);
    StringRef Delimiter = ClPrettyPrint ? ": " : "\n";
    outs() << Delimiter;
  }
  Offset -= ClAdjustVMA;
  object::SectionedAddress ModuleOffset = {
      Offset, getModuleSectionIndexForAddress(ModuleName, Offset)};
  if (IsData) {
    auto ResOrErr = Symbolizer.symbolizeData(ModuleName, ModuleOffset);
    Printer << (error(ResOrErr) ? DIGlobal() : ResOrErr.get());
  } else if (ClPrintInlining) {
    auto ResOrErr =
        Symbolizer.symbolizeInlinedCode(ModuleName, ModuleOffset, ClDwpName);
    Printer << (error(ResOrErr) ? DIInliningInfo() : ResOrErr.get());
  } else {
    auto ResOrErr =
        Symbolizer.symbolizeCode(ModuleName, ModuleOffset, ClDwpName);
    Printer << (error(ResOrErr) ? DILineInfo() : ResOrErr.get());
  }
  outs() << "\n";
  outs().flush();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);
  cl::ParseCommandLineOptions(argc, argv, "llvm-symbolizer\n");

  // If both --demangle and --no-demangle are specified then pick the last one.
  if (ClNoDemangle.getPosition() > ClDemangle.getPosition())
    ClDemangle = !ClNoDemangle;

  LLVMSymbolizer::Options Opts(ClPrintFunctions, ClUseSymbolTable, ClDemangle,
                               ClUseRelativeAddress, ClDefaultArch,
                               ClFallbackDebugPath);

  for (const auto &hint : ClDsymHint) {
    if (sys::path::extension(hint) == ".dSYM") {
      Opts.DsymHints.push_back(hint);
    } else {
      errs() << "Warning: invalid dSYM hint: \"" << hint <<
                "\" (must have the '.dSYM' extension).\n";
    }
  }
  LLVMSymbolizer Symbolizer(Opts);

  DIPrinter Printer(outs(), ClPrintFunctions != FunctionNameKind::None,
                    ClPrettyPrint, ClPrintSourceContextLines, ClVerbose,
                    ClBasenames);

  if (ClInputAddresses.empty()) {
    const int kMaxInputStringLength = 1024;
    char InputString[kMaxInputStringLength];

    while (fgets(InputString, sizeof(InputString), stdin))
      symbolizeInput(InputString, Symbolizer, Printer);
  } else {
    for (StringRef Address : ClInputAddresses)
      symbolizeInput(Address, Symbolizer, Printer);
  }

  return 0;
}
