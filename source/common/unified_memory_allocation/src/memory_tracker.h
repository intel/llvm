/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMA_MEMORY_TRACKER_INTERNAL_H
#define UMA_MEMORY_TRACKER_INTERNAL_H 1

#include <uma/base.h>
#include <uma/memory_pool.h>
#include <uma/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uma_memory_tracker_t *uma_memory_tracker_handle_t;

uma_memory_tracker_handle_t umaMemoryTrackerGet(void);
void *umaMemoryTrackerGetPool(uma_memory_tracker_handle_t hTracker,
                              const void *ptr);

// Creates a memory provider that tracks each allocation/deallocation through uma_memory_tracker_handle_t and
// forwards all requests to hUpstream memory Provider. hUpstream liftime should be managed by the user of this function.
enum uma_result_t umaTrackingMemoryProviderCreate(
    uma_memory_provider_handle_t hUpstream, uma_memory_pool_handle_t hPool,
    uma_memory_provider_handle_t *hTrackingProvider);

void umaTrackingMemoryProviderGetUpstreamProvider(
    uma_memory_provider_handle_t hTrackingProvider,
    uma_memory_provider_handle_t *hUpstream);

#ifdef __cplusplus
}
#endif

#endif /* UMA_MEMORY_TRACKER_INTERNAL_H */
