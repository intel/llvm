/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "helpers.h"

#include <fstream>
#include <sstream>

std::string generate_plus_one_spv() {
    std::ifstream spv(CURRENT_SOURCE_DIR "/asan.spv");
    std::stringstream buffer;
    buffer << spv.rdbuf();

    return buffer.str();
}
