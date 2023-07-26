/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "memory_pool_internal.h"
#include "memory_provider_internal.h"
#include "memory_tracker.h"

#include <umf/memory_pool.h>

#include <assert.h>
#include <stdlib.h>

static void
destroyMemoryProviderWrappers(umf_memory_provider_handle_t *providers,
                              size_t numProviders) {
    for (size_t i = 0; i < numProviders; i++) {
        umfMemoryProviderDestroy(providers[i]);
    }

    free(providers);
}

enum umf_result_t umfPoolCreate(const struct umf_memory_pool_ops_t *ops,
                                umf_memory_provider_handle_t *providers,
                                size_t numProviders, void *params,
                                umf_memory_pool_handle_t *hPool) {
    if (!numProviders || !providers) {
        return UMF_RESULT_ERROR_INVALID_ARGUMENT;
    }

    enum umf_result_t ret = UMF_RESULT_SUCCESS;
    umf_memory_pool_handle_t pool = malloc(sizeof(struct umf_memory_pool_t));
    if (!pool) {
        return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    assert(ops->version == UMF_VERSION_CURRENT);

    pool->providers =
        calloc(numProviders, sizeof(umf_memory_provider_handle_t));
    if (!pool->providers) {
        ret = UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        goto err_providers_alloc;
    }

    size_t providerInd = 0;
    pool->numProviders = numProviders;

    // Wrap each provider with memory tracking provider.
    for (providerInd = 0; providerInd < numProviders; providerInd++) {
        ret = umfTrackingMemoryProviderCreate(providers[providerInd], pool,
                                              &pool->providers[providerInd]);
        if (ret != UMF_RESULT_SUCCESS) {
            goto err_providers_init;
        }
    }

    pool->ops = *ops;
    ret = ops->initialize(pool->providers, pool->numProviders, params,
                          &pool->pool_priv);
    if (ret != UMF_RESULT_SUCCESS) {
        goto err_pool_init;
    }

    *hPool = pool;
    return UMF_RESULT_SUCCESS;

err_pool_init:
err_providers_init:
    destroyMemoryProviderWrappers(pool->providers, providerInd);
err_providers_alloc:
    free(pool);

    return ret;
}

void umfPoolDestroy(umf_memory_pool_handle_t hPool) {
    hPool->ops.finalize(hPool->pool_priv);
    destroyMemoryProviderWrappers(hPool->providers, hPool->numProviders);
    free(hPool);
}

enum umf_result_t umfFree(void *ptr) {
    umf_memory_pool_handle_t hPool = umfPoolByPtr(ptr);
    if (hPool) {
        return umfPoolFree(hPool, ptr);
    }
    return UMF_RESULT_SUCCESS;
}

umf_memory_pool_handle_t umfPoolByPtr(const void *ptr) {
    return umfMemoryTrackerGetPool(umfMemoryTrackerGet(), ptr);
}

enum umf_result_t
umfPoolGetMemoryProviders(umf_memory_pool_handle_t hPool, size_t numProviders,
                          umf_memory_provider_handle_t *hProviders,
                          size_t *numProvidersRet) {
    if (hProviders && numProviders < hPool->numProviders) {
        return UMF_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (numProvidersRet) {
        *numProvidersRet = hPool->numProviders;
    }

    if (hProviders) {
        for (size_t i = 0; i < hPool->numProviders; i++) {
            umfTrackingMemoryProviderGetUpstreamProvider(
                umfMemoryProviderGetPriv(hPool->providers[i]), hProviders + i);
        }
    }

    return UMF_RESULT_SUCCESS;
}
