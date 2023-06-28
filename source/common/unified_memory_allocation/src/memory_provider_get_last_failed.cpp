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

static thread_local uma_memory_provider_handle_t lastFailedProvider = nullptr;

uma_memory_provider_handle_t *umaGetLastFailedMemoryProviderPtr() {
    return &lastFailedProvider;
}
}
