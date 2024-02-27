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

#include <sstream>
#include <string>

namespace ur_sanitizer_layer {

bool ExtractSourceInfo(const std::string &output, SourceInfo &SI) {
    auto p1 = output.find('\n');
    std::string function = output.substr(0, p1);
    auto p2 = output.find(':', ++p1);
    std::string file = output.substr(p1, p2 - p1);
    auto p3 = output.find(':', ++p2);
    int line = std::stoi(output.substr(p2, p3 - p2));
    int column = std::stoi(output.substr(++p3));
    if (function != "??") {
        SI.function = std::move(function);
    }
    if (file != "??") {
        SI.file = std::move(file);
    }
    SI.line = line;
    SI.column = column;
    return true;
}

bool LLVMSymbolizer::SymbolizePC(const BacktraceInfo &BI, SourceInfo &SI) {
    std::stringstream ss;
    ss << "llvm-symbolizer --obj=" << BI.module << " " << BI.offset;
    auto result = RunCommand(ss.str().c_str());
    // context.logger.debug("llvm-symbolizer: {}", result);
    ExtractSourceInfo(result, SI);
    return true;
}

} // namespace ur_sanitizer_layer
