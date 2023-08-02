/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "memory_provider_internal.h"

extern "C" {

static thread_local umf_memory_provider_handle_t lastFailedProvider = nullptr;

umf_memory_provider_handle_t *umfGetLastFailedMemoryProviderPtr(void) {
    return &lastFailedProvider;
}
}
