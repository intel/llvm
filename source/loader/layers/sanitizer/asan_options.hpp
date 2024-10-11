/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_options.hpp
 *
 */

#pragma once

#include "common.hpp"

namespace ur_sanitizer_layer {

struct AsanOptions {
    bool Debug = false;
    uint64_t MinRZSize = 16;
    uint64_t MaxRZSize = 2048;
    uint32_t MaxQuarantineSizeMB = 0;
    bool DetectLocals = true;
    bool DetectPrivates = true;
    bool PrintStats = false;
    bool DetectKernelArguments = true;

    explicit AsanOptions();
};

} // namespace ur_sanitizer_layer
