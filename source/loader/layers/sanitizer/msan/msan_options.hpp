/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_options.hpp
 *
 */

#pragma once

#include <cstdint>

namespace ur_sanitizer_layer {
namespace msan {

struct MsanOptions {
    bool Debug = false;

    explicit MsanOptions();
};

} // namespace msan
} // namespace ur_sanitizer_layer
