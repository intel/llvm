// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple check to ensure the "maybe" macros select the correct L0 version

// RUN: %maybe-v1 ./confirm_version | FileCheck %s --check-prefix CHECK-V1
// RUN: %maybe-v2 ./confirm_version | FileCheck %s --check-prefix CHECK-V2

#include "ur_api.h"

#include <array>
#include <cassert>
#include <iostream>

#define CHECK(X)                                                               \
  if (X)                                                                       \
    return 1;

int main() {
  ur_result_t Err;

  Err = urLoaderInit(0, nullptr);

  ur_adapter_handle_t adapter;
  Err = urAdapterGet(1, &adapter, nullptr);
  CHECK(Err);

  uint32_t version;
  Err = urAdapterGetInfo(adapter, UR_ADAPTER_INFO_VERSION, sizeof(version),
                         &version, nullptr);
  CHECK(Err);

  // CHECK-V1: Adapter version: 1
  // CHECK-V2: Adapter version: 2
  std::cout << "Adapter version: " << version << "\n";

  return 0;
}
