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

#include <umf/memory_pool.h>

#include <assert.h>
#include <stdlib.h>

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

    for (providerInd = 0; providerInd < numProviders; providerInd++) {
        pool->providers[providerInd] = providers[providerInd];
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
    free(pool->providers);
err_providers_alloc:
    free(pool);

    return ret;
}

void umfPoolDestroy(umf_memory_pool_handle_t hPool) {
    hPool->ops.finalize(hPool->pool_priv);
    free(hPool->providers);
    free(hPool);
}

enum umf_result_t umfFree(void *ptr) {
    (void)ptr;
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
}

umf_memory_pool_handle_t umfPoolByPtr(const void *ptr) {
    (void)ptr;
    return NULL;
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
            hProviders[i] = hPool->providers[i];
        }
    }

    return UMF_RESULT_SUCCESS;
}
