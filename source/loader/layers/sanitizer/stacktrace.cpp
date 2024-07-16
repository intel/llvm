/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file stacktrace.cpp
 *
 */

#include "stacktrace.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

namespace {

bool Contains(const std::string &s, const char *p) {
    return s.find(p) != std::string::npos;
}

} // namespace

void StackTrace::print() const {
    if (!stack.size()) {
        getContext()->logger.always("  failed to acquire backtrace");
    }

    unsigned index = 0;

    for (auto &BI : stack) {
        // Skip runtime modules
        if (Contains(BI, "libsycl.so") ||
            Contains(BI, "libpi_unified_runtime.so") ||
            Contains(BI, "libur_loader.so")) {
            continue;
        }
        getContext()->logger.always("  #{} {}", index, BI);
        ++index;
    }
    getContext()->logger.always("");
}

} // namespace ur_sanitizer_layer
