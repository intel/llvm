/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "memory_tracker.h"
#include "critnib/critnib.h"

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/memory_provider_ops.h>

#include <assert.h>
#include <errno.h>
#include <stdlib.h>

#if !defined(_WIN32)
critnib *TRACKER = NULL;
void __attribute__((constructor)) createLibTracker(void) {
    TRACKER = critnib_new();
}
void __attribute__((destructor)) deleteLibTracker(void) {
    critnib_delete(TRACKER);
}

umf_memory_tracker_handle_t umfMemoryTrackerGet(void) {
    return (umf_memory_tracker_handle_t)TRACKER;
}
#endif

struct tracker_value_t {
    umf_memory_pool_handle_t pool;
    size_t size;
};

static enum umf_result_t
umfMemoryTrackerAdd(umf_memory_tracker_handle_t hTracker,
                    umf_memory_pool_handle_t pool, const void *ptr,
                    size_t size) {
    assert(ptr);

    struct tracker_value_t *value =
        (struct tracker_value_t *)malloc(sizeof(struct tracker_value_t));
    value->pool = pool;
    value->size = size;

    int ret = critnib_insert((critnib *)hTracker, (uintptr_t)ptr, value, 0);

    if (ret == 0) {
        return UMF_RESULT_SUCCESS;
    }

    free(value);

    if (ret == ENOMEM) {
        return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    // This should not happen
    // TODO: add logging here
    return UMF_RESULT_ERROR_UNKNOWN;
}

static enum umf_result_t
umfMemoryTrackerRemove(umf_memory_tracker_handle_t hTracker, const void *ptr,
                       size_t size) {
    assert(ptr);

    // TODO: there is no support for removing partial ranges (or multipe entires
    // in a single remove call) yet.
    // Every umfMemoryTrackerAdd(..., ptr, ...) should have a corresponsding
    // umfMemoryTrackerRemove call with the same ptr value.
    (void)size;

    void *value = critnib_remove((critnib *)hTracker, (uintptr_t)ptr);
    if (!value) {
        // This should not happen
        // TODO: add logging here
        return UMF_RESULT_ERROR_UNKNOWN;
    }

    free(value);

    return UMF_RESULT_SUCCESS;
}

umf_memory_pool_handle_t
umfMemoryTrackerGetPool(umf_memory_tracker_handle_t hTracker, const void *ptr) {
    assert(ptr);

    uintptr_t rkey;
    struct tracker_value_t *rvalue;
    int found = critnib_find((critnib *)hTracker, (uintptr_t)ptr, FIND_LE,
                             (void *)&rkey, (void **)&rvalue);
    if (!found) {
        return NULL;
    }

    return (rkey + rvalue->size >= (uintptr_t)ptr) ? rvalue->pool : NULL;
}

struct umf_tracking_memory_provider_t {
    umf_memory_provider_handle_t hUpstream;
    umf_memory_tracker_handle_t hTracker;
    umf_memory_pool_handle_t pool;
};

typedef struct umf_tracking_memory_provider_t umf_tracking_memory_provider_t;

static enum umf_result_t trackingAlloc(void *hProvider, size_t size,
                                       size_t alignment, void **ptr) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)hProvider;
    enum umf_result_t ret = UMF_RESULT_SUCCESS;

    if (!p->hUpstream) {
        return UMF_RESULT_ERROR_INVALID_ARGUMENT;
    }

    ret = umfMemoryProviderAlloc(p->hUpstream, size, alignment, ptr);
    if (ret != UMF_RESULT_SUCCESS || !*ptr) {
        return ret;
    }

    ret = umfMemoryTrackerAdd(p->hTracker, p->pool, *ptr, size);
    if (ret != UMF_RESULT_SUCCESS && p->hUpstream) {
        if (umfMemoryProviderFree(p->hUpstream, *ptr, size)) {
            // TODO: LOG
        }
    }

    return ret;
}

static enum umf_result_t trackingFree(void *hProvider, void *ptr, size_t size) {
    enum umf_result_t ret;
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)hProvider;

    // umfMemoryTrackerRemove should be called before umfMemoryProviderFree
    // to avoid a race condition. If the order would be different, other thread
    // could allocate the memory at address `ptr` before a call to umfMemoryTrackerRemove
    // resulting in inconsistent state.
    if (ptr) {
        ret = umfMemoryTrackerRemove(p->hTracker, ptr, size);
        if (ret != UMF_RESULT_SUCCESS) {
            return ret;
        }
    }

    ret = umfMemoryProviderFree(p->hUpstream, ptr, size);
    if (ret != UMF_RESULT_SUCCESS) {
        if (umfMemoryTrackerAdd(p->hTracker, p->pool, ptr, size) !=
            UMF_RESULT_SUCCESS) {
            // TODO: LOG
        }
        return ret;
    }

    return ret;
}

static enum umf_result_t trackingInitialize(void *params, void **ret) {
    umf_tracking_memory_provider_t *provider =
        (umf_tracking_memory_provider_t *)malloc(
            sizeof(umf_tracking_memory_provider_t));
    if (!provider) {
        return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *provider = *((umf_tracking_memory_provider_t *)params);
    *ret = provider;
    return UMF_RESULT_SUCCESS;
}

static void trackingFinalize(void *provider) { free(provider); }

static void trackingGetLastError(void *provider, const char **msg,
                                 int32_t *pError) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)provider;
    umfMemoryProviderGetLastNativeError(p->hUpstream, msg, pError);
}

static enum umf_result_t
trackingGetRecommendedPageSize(void *provider, size_t size, size_t *pageSize) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)provider;
    return umfMemoryProviderGetRecommendedPageSize(p->hUpstream, size,
                                                   pageSize);
}

static enum umf_result_t trackingGetMinPageSize(void *provider, void *ptr,
                                                size_t *pageSize) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)provider;
    return umfMemoryProviderGetMinPageSize(p->hUpstream, ptr, pageSize);
}

static enum umf_result_t trackingPurgeLazy(void *provider, void *ptr,
                                           size_t size) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)provider;
    return umfMemoryProviderPurgeLazy(p->hUpstream, ptr, size);
}

static enum umf_result_t trackingPurgeForce(void *provider, void *ptr,
                                            size_t size) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)provider;
    return umfMemoryProviderPurgeForce(p->hUpstream, ptr, size);
}

static const char *trackingName(void *provider) {
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)provider;
    return umfMemoryProviderGetName(p->hUpstream);
}

enum umf_result_t umfTrackingMemoryProviderCreate(
    umf_memory_provider_handle_t hUpstream, umf_memory_pool_handle_t hPool,
    umf_memory_provider_handle_t *hTrackingProvider) {
    umf_tracking_memory_provider_t params;
    params.hUpstream = hUpstream;
    params.hTracker = umfMemoryTrackerGet();
    params.pool = hPool;

    struct umf_memory_provider_ops_t trackingMemoryProviderOps;
    trackingMemoryProviderOps.version = UMF_VERSION_CURRENT;
    trackingMemoryProviderOps.initialize = trackingInitialize;
    trackingMemoryProviderOps.finalize = trackingFinalize;
    trackingMemoryProviderOps.alloc = trackingAlloc;
    trackingMemoryProviderOps.free = trackingFree;
    trackingMemoryProviderOps.get_last_native_error = trackingGetLastError;
    trackingMemoryProviderOps.get_min_page_size = trackingGetMinPageSize;
    trackingMemoryProviderOps.get_recommended_page_size =
        trackingGetRecommendedPageSize;
    trackingMemoryProviderOps.purge_force = trackingPurgeForce;
    trackingMemoryProviderOps.purge_lazy = trackingPurgeLazy;
    trackingMemoryProviderOps.get_name = trackingName;

    return umfMemoryProviderCreate(&trackingMemoryProviderOps, &params,
                                   hTrackingProvider);
}

void umfTrackingMemoryProviderGetUpstreamProvider(
    umf_memory_provider_handle_t hTrackingProvider,
    umf_memory_provider_handle_t *hUpstream) {
    assert(hUpstream);
    umf_tracking_memory_provider_t *p =
        (umf_tracking_memory_provider_t *)hTrackingProvider;
    *hUpstream = p->hUpstream;
}
