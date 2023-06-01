/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMA_MEMORY_PROVIDER_INTERNAL_H
#define UMA_MEMORY_PROVIDER_INTERNAL_H 1

#include <uma/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

void *umaMemoryProviderGetPriv(uma_memory_provider_handle_t hProvider);

#ifdef __cplusplus
}
#endif

#endif /* UMA_MEMORY_PROVIDER_INTERNAL_H */
