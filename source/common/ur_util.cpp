/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "ur_util.hpp"

#ifdef _WIN32
#include <windows.h>
int ur_getpid(void) { return static_cast<int>(GetCurrentProcessId()); }
#else

#include <unistd.h>
int ur_getpid(void) { return static_cast<int>(getpid()); }
#endif

std::optional<std::string> ur_getenv(const char *name) {
#if defined(_WIN32)
    constexpr int buffer_size = 1024;
    char buffer[buffer_size];
    auto rc = GetEnvironmentVariableA(name, buffer, buffer_size);
    if (0 != rc && rc < buffer_size) {
        return std::string(buffer);
    } else if (rc >= buffer_size) {
        std::stringstream ex_ss;
        ex_ss << "Environment variable " << name << " value too long!"
              << " Maximum length is " << buffer_size - 1 << " characters.";
        throw std::invalid_argument(ex_ss.str());
    }
    return std::nullopt;
#else
    const char *tmp_env = getenv(name);
    if (tmp_env != nullptr) {
        return std::string(tmp_env);
    } else {
        return std::nullopt;
    }
#endif
}
