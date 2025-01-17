/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file ur_codeloc.hpp
 *
 */

#ifndef UR_CODELOC_HPP
#define UR_CODELOC_HPP 1

#include "ur_api.h"
#include <optional>

struct codeloc_data {
  codeloc_data() {
    codelocCb = nullptr;
    codelocUserdata = nullptr;
  }
  ur_code_location_callback_t codelocCb;
  void *codelocUserdata;

  std::optional<ur_code_location_t> get_codeloc() {
    if (!codelocCb) {
      return std::nullopt;
    }
    return codelocCb(codelocUserdata);
  }
};

#endif /* UR_CODELOC_HPP */
