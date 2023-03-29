/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include "backtrace.hpp"

#include <backtrace.h>
#include <cxxabi.h>
#include <limits.h>
#include <vector>

namespace validation_layer {

int backtrace_cb(void *data, uintptr_t pc, const char *filename, int lineno,
                 const char *function) {
    if (filename == NULL && function == NULL) {
        return 0;
    }

    std::stringstream backtraceLine;
    backtraceLine << "0x" << std::hex << pc << " in ";

    int status;
    char *demangled = abi::__cxa_demangle(function, NULL, NULL, &status);
    if (status == 0) {
        backtraceLine << "(" << demangled << ") ";
    } else if (function != NULL) {
        backtraceLine << "(" << function << ") ";
    } else {
        backtraceLine << "(????????) ";
    }

    char filepath[PATH_MAX];
    if (realpath(filename, filepath) != NULL) {
        backtraceLine << "(" << filepath << ":" << std::dec << lineno << ")";
    } else {
        backtraceLine << "(????????)";
    }

    std::vector<std::string> *backtrace =
        reinterpret_cast<std::vector<std::string> *>(data);
    try {
        if (backtraceLine.str().empty()) {
            backtrace->push_back("????????");
        } else {
            backtrace->push_back(backtraceLine.str());
        }
    } catch (std::bad_alloc &) {
    }

    free(demangled);

    return 0;
}

std::vector<std::string> getCurrentBacktrace() {
    backtrace_state *state = backtrace_create_state(NULL, 0, NULL, NULL);
    if (state == NULL) {
        return std::vector<std::string>(1, "Failed to acquire a backtrace");
    }

    std::vector<std::string> backtrace;
    backtrace_full(state, 0, backtrace_cb, NULL, &backtrace);
    if (backtrace.empty()) {
        return std::vector<std::string>(1, "Failed to acquire a backtrace");
    }

    return backtrace;
}

} // namespace validation_layer
