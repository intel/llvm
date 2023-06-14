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
#include <uma/memory_provider.h>
#include <uma/memory_provider_ops.h>

#include <cassert>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdlib.h>

// TODO: reimplement in C and optimize...
struct uma_memory_tracker_t {
    enum uma_result_t add(void *pool, const void *ptr, size_t size) {
        std::unique_lock<std::shared_mutex> lock(mtx);

        if (size == 0) {
            return UMA_RESULT_SUCCESS;
        }

        auto ret =
            map.try_emplace(reinterpret_cast<uintptr_t>(ptr), size, pool);
        return ret.second ? UMA_RESULT_SUCCESS : UMA_RESULT_ERROR_UNKNOWN;
    }

    enum uma_result_t remove(const void *ptr, size_t size) {
        std::unique_lock<std::shared_mutex> lock(mtx);

        map.erase(reinterpret_cast<uintptr_t>(ptr));

        // TODO: handle removing part of the range
        (void)size;

        return UMA_RESULT_SUCCESS;
    }

    void *find(const void *ptr) {
        std::shared_lock<std::shared_mutex> lock(mtx);

        auto intptr = reinterpret_cast<uintptr_t>(ptr);
        auto it = map.upper_bound(intptr);
        if (it == map.begin()) {
            return nullptr;
        }

        --it;

        auto address = it->first;
        auto size = it->second.first;
        auto pool = it->second.second;

        if (intptr >= address && intptr < address + size) {
            return pool;
        }

        return nullptr;
    }

  private:
    std::shared_mutex mtx;
    std::map<uintptr_t, std::pair<size_t, void *>> map;
};

static enum uma_result_t
umaMemoryTrackerAdd(uma_memory_tracker_handle_t hTracker, void *pool,
                    const void *ptr, size_t size) {
    return hTracker->add(pool, ptr, size);
}

static enum uma_result_t
umaMemoryTrackerRemove(uma_memory_tracker_handle_t hTracker, const void *ptr,
                       size_t size) {
    return hTracker->remove(ptr, size);
}

extern "C" {

uma_memory_tracker_handle_t umaMemoryTrackerGet(void) {
    static uma_memory_tracker_t tracker;
    return &tracker;
}

void *umaMemoryTrackerGetPool(uma_memory_tracker_handle_t hTracker,
                              const void *ptr) {
    return hTracker->find(ptr);
}

struct uma_tracking_memory_provider_t {
    uma_memory_provider_handle_t hUpstream;
    uma_memory_tracker_handle_t hTracker;
    uma_memory_pool_handle_t pool;
};

typedef struct uma_tracking_memory_provider_t uma_tracking_memory_provider_t;

static enum uma_result_t trackingAlloc(void *hProvider, size_t size,
                                       size_t alignment, void **ptr) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)hProvider;
    enum uma_result_t ret = UMA_RESULT_SUCCESS;

    if (!p->hUpstream) {
        return UMA_RESULT_ERROR_INVALID_ARGUMENT;
    }

    ret = umaMemoryProviderAlloc(p->hUpstream, size, alignment, ptr);
    if (ret != UMA_RESULT_SUCCESS) {
        return ret;
    }

    ret = umaMemoryTrackerAdd(p->hTracker, p->pool, *ptr, size);
    if (ret != UMA_RESULT_SUCCESS && p->hUpstream) {
        if (umaMemoryProviderFree(p->hUpstream, *ptr, size)) {
            // TODO: LOG
        }
    }

    return ret;
}

static enum uma_result_t trackingFree(void *hProvider, void *ptr, size_t size) {
    enum uma_result_t ret;
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)hProvider;

    // umaMemoryTrackerRemove should be called before umaMemoryProviderFree
    // to avoid a race condition. If the order would be different, other thread
    // could allocate the memory at address `ptr` before a call to umaMemoryTrackerRemove
    // resulting in inconsistent state.
    ret = umaMemoryTrackerRemove(p->hTracker, ptr, size);
    if (ret != UMA_RESULT_SUCCESS) {
        return ret;
    }

    ret = umaMemoryProviderFree(p->hUpstream, ptr, size);
    if (ret != UMA_RESULT_SUCCESS) {
        if (umaMemoryTrackerAdd(p->hTracker, p->pool, ptr, size) !=
            UMA_RESULT_SUCCESS) {
            // TODO: LOG
        }
        return ret;
    }

    return ret;
}

static enum uma_result_t trackingInitialize(void *params, void **ret) {
    uma_tracking_memory_provider_t *provider =
        (uma_tracking_memory_provider_t *)malloc(
            sizeof(uma_tracking_memory_provider_t));
    if (!provider) {
        return UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *provider = *((uma_tracking_memory_provider_t *)params);
    *ret = provider;
    return UMA_RESULT_SUCCESS;
}

static void trackingFinalize(void *provider) { free(provider); }

static enum uma_result_t trackingGetLastResult(void *provider,
                                               const char **msg) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)provider;
    return umaMemoryProviderGetLastResult(p->hUpstream, msg);
}

static enum uma_result_t
trackingGetRecommendedPageSize(void *provider, size_t size, size_t *pageSize) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)provider;
    return umaMemoryProviderGetRecommendedPageSize(p->hUpstream, size,
                                                   pageSize);
}

static enum uma_result_t trackingGetMinPageSize(void *provider, void *ptr,
                                                size_t *pageSize) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)provider;
    return umaMemoryProviderGetMinPageSize(p->hUpstream, ptr, pageSize);
}

static enum uma_result_t trackingPurgeLazy(void *provider, void *ptr,
                                           size_t size) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)provider;
    return umaMemoryProviderPurgeLazy(p->hUpstream, ptr, size);
}

static enum uma_result_t trackingPurgeForce(void *provider, void *ptr,
                                            size_t size) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)provider;
    return umaMemoryProviderPurgeForce(p->hUpstream, ptr, size);
}

static void trackingName(void *provider, const char **ppName) {
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)provider;
    return umaMemoryProviderGetName(p->hUpstream, ppName);
}

enum uma_result_t umaTrackingMemoryProviderCreate(
    uma_memory_provider_handle_t hUpstream, uma_memory_pool_handle_t hPool,
    uma_memory_provider_handle_t *hTrackingProvider) {
    uma_tracking_memory_provider_t params;
    params.hUpstream = hUpstream;
    params.hTracker = umaMemoryTrackerGet();
    params.pool = hPool;

    struct uma_memory_provider_ops_t trackingMemoryProviderOps;
    trackingMemoryProviderOps.version = UMA_VERSION_CURRENT;
    trackingMemoryProviderOps.initialize = trackingInitialize;
    trackingMemoryProviderOps.finalize = trackingFinalize;
    trackingMemoryProviderOps.alloc = trackingAlloc;
    trackingMemoryProviderOps.free = trackingFree;
    trackingMemoryProviderOps.get_last_result = trackingGetLastResult;
    trackingMemoryProviderOps.get_min_page_size = trackingGetMinPageSize;
    trackingMemoryProviderOps.get_recommended_page_size =
        trackingGetRecommendedPageSize;
    trackingMemoryProviderOps.purge_force = trackingPurgeForce;
    trackingMemoryProviderOps.purge_lazy = trackingPurgeLazy;
    trackingMemoryProviderOps.get_name = trackingName;

    return umaMemoryProviderCreate(&trackingMemoryProviderOps, &params,
                                   hTrackingProvider);
}

void umaTrackingMemoryProviderGetUpstreamProvider(
    uma_memory_provider_handle_t hTrackingProvider,
    uma_memory_provider_handle_t *hUpstream) {
    assert(hUpstream);
    uma_tracking_memory_provider_t *p =
        (uma_tracking_memory_provider_t *)hTrackingProvider;
    *hUpstream = p->hUpstream;
}
}
