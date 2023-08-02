/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_PROVIDER_INTERNAL_H
#define UMF_MEMORY_PROVIDER_INTERNAL_H 1

#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

void *umfMemoryProviderGetPriv(umf_memory_provider_handle_t hProvider);
umf_memory_provider_handle_t *umfGetLastFailedMemoryProviderPtr(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_PROVIDER_INTERNAL_H */
