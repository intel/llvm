//===--- sanitizer_stacktrace.cpp - Host stack traces ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_stacktrace.hpp"
#include "sanitizer_log.hpp"

#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"

#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <execinfo.h>
#include <link.h>

namespace llvm::offload::sanitizer {

namespace {

bool Contains(const std::string &s, const char *p) {
  return s.find(p) != std::string::npos;
}

// Parse back trace information in the following formats:
//   <module_name>([function_name]+function_offset) [offset]
void ParseBacktraceInfo(const BacktraceInfo &BI, std::string &ModuleName,
                        uptr &Offset) {
  // Parse module name
  size_t End = BI.find_first_of('(');
  assert(End != std::string::npos);
  ModuleName = BI.substr(0, End);
  // Parse offset
  size_t Start = BI.find_first_of('[');
  assert(Start != std::string::npos);
  Start = BI.substr(Start + 1, 2) == "0x" ? Start + 3 : Start + 1;
  End = BI.find_first_of(']');
  assert(End != std::string::npos);
  Offset = std::stoull(BI.substr(Start, End), nullptr, 16);
}

// Parse symbolizer output in the following formats:
//   <function_name>
//   <file_name>:<line_number>[:<column_number>]
SourceInfo ParseSymbolizerOutput(const std::string &Output) {
  SourceInfo Info;
  // Parse function name
  size_t End = Output.find_first_of('\n');
  assert(End != std::string::npos);
  Info.function = Output.substr(0, End);
  // Parse file name
  size_t Start = End + 1;
  End = Output.find_first_of(':', Start);
  assert(End != std::string::npos);
  Info.file = Output.substr(Start, End - Start);
  // Parse line number
  Start = End + 1;
  End = Output.find_first_of(":\n", Start);
  assert(End != std::string::npos);
  Info.line = std::stoi(Output.substr(Start, End - Start));
  // Parse column number if exists
  if (Output[End] == ':') {
    Start = End + 1;
    End = Output.find_first_of("\n", Start);
    assert(End != std::string::npos);
    Info.column = std::stoi(Output.substr(Start, End - Start));
  }

  return Info;
}

uintptr_t GetModuleBase(const char *ModuleName) {
  uintptr_t Data = (uintptr_t)ModuleName;
  int Result = dl_iterate_phdr(
      [](struct dl_phdr_info *Info, size_t, void *Arg) {
        uintptr_t *Data = (uintptr_t *)Arg;
        const char *ModuleName = (const char *)(*Data);
        if (strstr(Info->dlpi_name, ModuleName)) {
          *Data = (uintptr_t)Info->dlpi_addr;
          return 1;
        }
        return 0;
      },
      (void *)&Data);

  // If dl_iterate_phdr return 0, it means the module is main executable,
  // its base address should be 0.
  if (!Result) {
    return 0;
  }
  return Data;
}

// Empty if the module can't be symbolized.
std::string SymbolizeCode(symbolize::LLVMSymbolizer &Symbolizer,
                          const std::string &ModuleName,
                          uint64_t ModuleOffset) {
  std::string Result;
  raw_string_ostream OS(Result);
  symbolize::Request Request{ModuleName, ModuleOffset};
  symbolize::PrinterConfig Config;
  Config.Pretty = false;
  Config.PrintAddress = false;
  Config.PrintFunctions = true;
  Config.SourceContextLines = 0;
  Config.Verbose = false;
  symbolize::ErrorHandler EH = [&](const ErrorInfoBase &ErrorInfo,
                                   StringRef ErrorBanner) {
    OS << ErrorBanner;
    ErrorInfo.log(OS);
    OS << '\n';
  };
  symbolize::LLVMPrinter Printer(OS, EH, Config);

  auto ResOrErr = Symbolizer.symbolizeInlinedCode(
      ModuleName, {ModuleOffset - GetModuleBase(ModuleName.c_str()),
                   object::SectionedAddress::UndefSection});
  if (!ResOrErr) {
    consumeError(ResOrErr.takeError());
    return {};
  }
  Printer.print(Request, *ResOrErr);
  Symbolizer.pruneCache();
  return Result;
}

} // namespace

void StackTrace::print() const {
  if (!stack.size()) {
    SANITIZER_REPORT("  failed to acquire backtrace");
    return;
  }

  unsigned index = 0;

  char **BacktraceSymbols = GetBacktraceSymbols(stack);
  symbolize::LLVMSymbolizer Symbolizer;

  for (size_t i = 0; i < stack.size(); i++) {
    BacktraceInfo BI = BacktraceSymbols[i];

    // Skip runtime modules
    if (Contains(BI, "libsycl.so") || Contains(BI, "libur_loader.so") ||
        Contains(BI, "libomptarget.rtl.unified_runtime.so") ||
        Contains(BI, "libomptarget.so") || Contains(BI, "libLLVMOffload.so") ||
        Contains(BI, "libur_adapter_offload.so") ||
        Contains(BI, "liboffload_msan_ze")) {
      continue;
    }

    std::string ModuleName;
    uptr Offset;
    ParseBacktraceInfo(BI, ModuleName, Offset);
    std::string Result = SymbolizeCode(Symbolizer, ModuleName, Offset);
    if (!Result.empty()) {
      SourceInfo SrcInfo = ParseSymbolizerOutput(Result);
      if (SrcInfo.file != "??") {
        SANITIZER_REPORT(" #{} in {} {}:{}:{}", index, SrcInfo.function,
                         SrcInfo.file, SrcInfo.line, SrcInfo.column);
      } else {
        SANITIZER_REPORT(" #{} in {} ({}+{})", index, SrcInfo.function,
                         ModuleName, (void *)Offset);
      }
    }
    ++index;
  }
  SANITIZER_REPORT("");

  free(BacktraceSymbols);
}

StackTrace GetCurrentBacktrace() {
  BacktraceFrame Frames[MAX_BACKTRACE_FRAMES];
  int FrameCount = backtrace(Frames, MAX_BACKTRACE_FRAMES);

  // The Frames contain the return addresses, which is one instruction after the
  // call instruction. Adjust the addresses so that symbolizer would give more
  // precise result.
  for (int I = 0; I < FrameCount; I++) {
    Frames[I] = (void *)((uintptr_t)Frames[I] - 1);
  }

  StackTrace Stack;
  Stack.stack = std::vector<BacktraceFrame>(&Frames[0], &Frames[FrameCount]);

  return Stack;
}

char **GetBacktraceSymbols(const std::vector<BacktraceFrame> &BacktraceFrames) {
  assert(!BacktraceFrames.empty());

  char **BacktraceSymbols =
      backtrace_symbols(&BacktraceFrames[0], BacktraceFrames.size());
  return BacktraceSymbols;
}

std::string DemangleName(const std::string &name) {
  std::string result = name;
  char *demangled =
      abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr);
  if (demangled) {
    result = demangled;
    free(demangled);
  }
  return result;
}

} // namespace llvm::offload::sanitizer
