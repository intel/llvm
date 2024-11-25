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

#include <cstdint>

namespace ur_sanitizer_layer {
namespace asan {

struct AsanOptions {
    bool Debug = false;
    uint64_t MinRZSize = 16;
    uint64_t MaxRZSize = 2048;
    uint32_t MaxQuarantineSizeMB = 8;
    bool DetectLocals = true;
    bool DetectPrivates = true;
    bool PrintStats = false;
    bool DetectKernelArguments = true;
    bool DetectLeaks = true;

    explicit AsanOptions();
};

} // namespace asan
} // namespace ur_sanitizer_layer
