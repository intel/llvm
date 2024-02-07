/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file symbolizer_llvm.cpp
 *
 */

#include "symbolizer_llvm.hpp"
#include "ur_sanitizer_layer.hpp"

#include <sstream>
#include <string>

namespace ur_sanitizer_layer {

std::string exec(const char *cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

bool ExtractInfo(const std::string &output, AddressInfo *Info) {
    auto p1 = output.find('\n');
    std::string function = output.substr(0, p1);
    auto p2 = output.find(':', ++p1);
    std::string file = output.substr(p1, p2 - p1);
    auto p3 = output.find(':', ++p2);
    int line = std::stoi(output.substr(p2, p3 - p2));
    int column = std::stoi(output.substr(++p3));
    if (function != "??") {
        Info->function = std::move(function);
    }
    if (file != "??") {
        Info->file = std::move(file);
    }
    Info->line = line;
    Info->column = column;
    return true;
}

LLVMSymbolizer::LLVMSymbolizer(const char *path) {}

bool LLVMSymbolizer::SymbolizePC(AddressInfo *Info) {
    std::stringstream ss;
    ss << "llvm-symbolizer --obj=" << Info->module << " " << Info->offset;
    auto result = exec(ss.str().c_str());
    // context.logger.debug("llvm-symbolizer: {}", result);
    ExtractInfo(result, Info);
    return true;
}

} // namespace ur_sanitizer_layer
