/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
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
