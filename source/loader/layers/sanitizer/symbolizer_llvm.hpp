/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file symbolizer_llvm.hpp
 *
 */

#pragma once

#include "common.hpp"

#include <memory>
#include <vector>

namespace ur_sanitizer_layer {

class SymbolizerTool {
  public:
    virtual ~SymbolizerTool() {}
    virtual bool symbolizePC(const BacktraceInfo &BI, SourceInfo &SI) {
        std::ignore = BI;
        std::ignore = SI;
        return false;
    }
};

class LLVMSymbolizer final : public SymbolizerTool {
  public:
    bool symbolizePC(const BacktraceInfo &BI, SourceInfo &SI) override;
};

extern std::vector<std::unique_ptr<SymbolizerTool>> SymbolizerTools;
void InitSymbolizers();

} // namespace ur_sanitizer_layer
