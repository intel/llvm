/*
 *
 * Copyright (C) 2023 Intel Corporation
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

namespace ur_sanitizer_layer {

class SymbolizerTool {
    virtual bool SymbolizePC(AddressInfo *Info) { return false; }
};

class LLVMSymbolizer final : public SymbolizerTool {
  public:
    explicit LLVMSymbolizer(const char *path);

    bool SymbolizePC(AddressInfo *Info) override;
};

} // namespace ur_sanitizer_layer
