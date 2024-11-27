/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file symbolizer.cpp
 *
 */
#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include <link.h>

static llvm::symbolize::PrinterConfig GetPrinterConfig() {
    llvm::symbolize::PrinterConfig Config;
    Config.Pretty = false;
    Config.PrintAddress = false;
    Config.PrintFunctions = true;
    Config.SourceContextLines = 0;
    Config.Verbose = false;
    return Config;
}

static uintptr_t GetModuleBase(const char *ModuleName) {
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

extern "C" {

void SymbolizeCode(const char *ModuleName, uint64_t ModuleOffset,
                   char *ResultString, size_t ResultSize, size_t *RetSize) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    llvm::symbolize::Request Request{ModuleName, ModuleOffset};
    llvm::symbolize::PrinterConfig Config = GetPrinterConfig();
    llvm::symbolize::ErrorHandler EH = [&](const llvm::ErrorInfoBase &ErrorInfo,
                                           llvm::StringRef ErrorBanner) {
        OS << ErrorBanner;
        ErrorInfo.log(OS);
        OS << '\n';
    };
    llvm::symbolize::LLVMSymbolizer Symbolizer;
    llvm::symbolize::LLVMPrinter Printer(OS, EH, Config);

    auto ResOrErr = Symbolizer.symbolizeInlinedCode(
        ModuleName, {ModuleOffset - GetModuleBase(ModuleName),
                     llvm::object::SectionedAddress::UndefSection});

    if (!ResOrErr) {
        return;
    }
    Printer.print(Request, *ResOrErr);
    Symbolizer.pruneCache();
    if (RetSize) {
        *RetSize = Result.size() + 1;
    }
    if (ResultString) {
        std::strncpy(ResultString, Result.c_str(), ResultSize);
        ResultString[ResultSize - 1] = '\0';
    }
}
}
