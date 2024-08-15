/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"

namespace ur_sanitizer_layer {

llvm::symbolize::LLVMSymbolizer *GetSymbolizer() {
    static llvm::symbolize::LLVMSymbolizer Symbolizer;
    return &Symbolizer;
}

llvm::symbolize::PrinterConfig GetPrinterConfig() {
    llvm::symbolize::PrinterConfig Config;
    Config.Pretty = false;
    Config.PrintAddress = false;
    Config.PrintFunctions = true;
    Config.SourceContextLines = 0;
    Config.Verbose = false;
    return Config;
}

} // namespace ur_sanitizer_layer

extern "C" {

bool SymbolizeCode(const std::string ModuleName, uint64_t ModuleOffset,
                   std::string &Result) {
    llvm::raw_string_ostream OS(Result);
    llvm::symbolize::Request Request{ModuleName, ModuleOffset};
    llvm::symbolize::PrinterConfig Config =
        ur_sanitizer_layer::GetPrinterConfig();
    llvm::symbolize::ErrorHandler EH = [&](const llvm::ErrorInfoBase &ErrorInfo,
                                           llvm::StringRef ErrorBanner) {
        OS << ErrorBanner;
        ErrorInfo.log(OS);
        OS << '\n';
    };
    auto Printer =
        std::make_unique<llvm::symbolize::LLVMPrinter>(OS, EH, Config);

    auto ResOrErr = ur_sanitizer_layer::GetSymbolizer()->symbolizeInlinedCode(
        ModuleName,
        {ModuleOffset, llvm::object::SectionedAddress::UndefSection});

    if (!ResOrErr) {
        return false;
    }
    Printer->print(Request, *ResOrErr);
    ur_sanitizer_layer::GetSymbolizer()->pruneCache();
    return true;
}
}
