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
#include <uma/memory_provider.h>

#include <assert.h>
#include <stdlib.h>

struct uma_memory_provider_t {
    struct uma_memory_provider_ops_t ops;
    void *provider_priv;
};

enum uma_result_t
umaMemoryProviderCreate(struct uma_memory_provider_ops_t *ops, void *params,
                        uma_memory_provider_handle_t *hProvider) {
    uma_memory_provider_handle_t provider =
        malloc(sizeof(struct uma_memory_provider_t));
    if (!provider) {
        return UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    assert(ops->version == UMA_VERSION_CURRENT);

    provider->ops = *ops;

    void *provider_priv;
    enum uma_result_t ret = ops->initialize(params, &provider_priv);
    if (ret != UMA_RESULT_SUCCESS) {
        free(provider);
        return ret;
    }

    provider->provider_priv = provider_priv;

    *hProvider = provider;

    return UMA_RESULT_SUCCESS;
}

void umaMemoryProviderDestroy(uma_memory_provider_handle_t hProvider) {
    hProvider->ops.finalize(hProvider->provider_priv);
    free(hProvider);
}

enum uma_result_t umaMemoryProviderAlloc(uma_memory_provider_handle_t hProvider,
                                         size_t size, size_t alignment,
                                         void **ptr) {
    return hProvider->ops.alloc(hProvider->provider_priv, size, alignment, ptr);
}

enum uma_result_t umaMemoryProviderFree(uma_memory_provider_handle_t hProvider,
                                        void *ptr, size_t size) {
    return hProvider->ops.free(hProvider->provider_priv, ptr, size);
}

enum uma_result_t
umaMemoryProviderGetLastResult(uma_memory_provider_handle_t hProvider,
                               const char **ppMessage) {
    return hProvider->ops.get_last_result(hProvider->provider_priv, ppMessage);
}

void *umaMemoryProviderGetPriv(uma_memory_provider_handle_t hProvider) {
    return hProvider->provider_priv;
}

enum uma_result_t
umaMemoryProviderGetRecommendedPageSize(uma_memory_provider_handle_t hProvider,
                                        size_t size, size_t *pageSize) {
    return hProvider->ops.get_recommended_page_size(hProvider->provider_priv,
                                                    size, pageSize);
}

enum uma_result_t
umaMemoryProviderGetMinPageSize(uma_memory_provider_handle_t hProvider,
                                void *ptr, size_t *pageSize) {
    return hProvider->ops.get_min_page_size(hProvider->provider_priv, ptr,
                                            pageSize);
}

enum uma_result_t
umaMemoryProviderPurgeLazy(uma_memory_provider_handle_t hProvider, void *ptr,
                           size_t size) {
    return hProvider->ops.purge_lazy(hProvider->provider_priv, ptr, size);
}

enum uma_result_t
umaMemoryProviderPurgeForce(uma_memory_provider_handle_t hProvider, void *ptr,
                            size_t size) {
    return hProvider->ops.purge_force(hProvider->provider_priv, ptr, size);
}

void umaMemoryProviderGetName(uma_memory_provider_handle_t hProvider,
                              const char **ppName) {
    hProvider->ops.get_name(hProvider->provider_priv, ppName);
}
