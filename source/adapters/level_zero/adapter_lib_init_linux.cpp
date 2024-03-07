/*
*
* Copyright (C) 2024 Intel Corporation
*
* Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
* See LICENSE.TXT
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*
* @file adapter_lib_init_linux.cpp
*
*/

#include "adapter.hpp"
#include "ur_level_zero.hpp"

void __attribute__((constructor)) createAdapterHandle() {
  if (!Adapter) {
    Adapter = new ur_adapter_handle_t_();
  }
}

void __attribute__((destructor)) deleteAdapterHandle() {
  if (Adapter) {
    delete Adapter;
  }
}
