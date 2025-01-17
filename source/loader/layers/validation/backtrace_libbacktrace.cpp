/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 */
#include "backtrace.hpp"

#include <backtrace.h>
#include <cxxabi.h>
#include <limits.h>
#include <vector>

namespace ur_validation_layer {

void filter_after_occurence(std::vector<BacktraceLine> &backtrace,
                            std::string substring) {
  auto it = std::find_if(backtrace.begin(), backtrace.end(),
                         [&substring](const std::string &s) {
                           return s.find(substring) != std::string::npos;
                         });

  if (it != backtrace.end()) {
    backtrace.erase(backtrace.begin(), it);
  }
}

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
    // Note: Escaping the last '?' character to avoid creation of a trigraph
    // character
    backtraceLine << "(???????\?) ";
  }

  char filepath[PATH_MAX];
  if (realpath(filename, filepath) != NULL) {
    backtraceLine << "(" << filepath << ":" << std::dec << lineno << ")";
  } else {
    // Note: Escaping the last '?' character to avoid creation of a trigraph
    // character
    backtraceLine << "(???????\?)";
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

std::vector<BacktraceLine> getCurrentBacktrace() {
  backtrace_state *state = backtrace_create_state(NULL, 0, NULL, NULL);
  if (state == NULL) {
    return std::vector<std::string>(1, "Failed to acquire a backtrace");
  }

  std::vector<BacktraceLine> backtrace;
  backtrace_full(state, 0, backtrace_cb, NULL, &backtrace);
  if (backtrace.empty()) {
    return std::vector<std::string>(1, "Failed to acquire a backtrace");
  }

  filter_after_occurence(backtrace, "ur_libapi.cpp");

  return backtrace;
}

} // namespace ur_validation_layer
