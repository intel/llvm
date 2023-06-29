// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider.h"

#include <uma/memory_provider_ops.h>

#include <assert.h>
#include <stdlib.h>

static enum uma_result_t nullInitialize(void *params, void **pool) {
    (void)params;
    *pool = NULL;
    return UMA_RESULT_SUCCESS;
}

static void nullFinalize(void *pool) { (void)pool; }

static enum uma_result_t nullAlloc(void *provider, size_t size,
                                   size_t alignment, void **ptr) {
    (void)provider;
    (void)size;
    (void)alignment;
    (void)ptr;
    return UMA_RESULT_SUCCESS;
}

static enum uma_result_t nullFree(void *provider, void *ptr, size_t size) {
    (void)provider;
    (void)ptr;
    (void)size;
    return UMA_RESULT_SUCCESS;
}

static enum uma_result_t nullGetLastResult(void *provider, const char **ppMsg) {
    (void)provider;
    (void)ppMsg;
    return UMA_RESULT_SUCCESS;
}

static enum uma_result_t nullGetRecommendedPageSize(void *provider, size_t size,
                                                    size_t *pageSize) {
    (void)provider;
    (void)size;
    (void)pageSize;
    return UMA_RESULT_SUCCESS;
}

static enum uma_result_t nullGetPageSize(void *provider, void *ptr,

                                         size_t *pageSize) {
    (void)provider;
    (void)ptr;
    (void)pageSize;
    return UMA_RESULT_SUCCESS;
}

static enum uma_result_t nullPurgeLazy(void *provider, void *ptr, size_t size) {
    (void)provider;
    (void)ptr;
    (void)size;
    return UMA_RESULT_SUCCESS;
}

static enum uma_result_t nullPurgeForce(void *provider, void *ptr,
                                        size_t size) {
    (void)provider;
    (void)ptr;
    (void)size;
    return UMA_RESULT_SUCCESS;
}

static void nullName(void *provider, const char **ppName) {
    (void)provider;
    *ppName = "null";
}

uma_memory_provider_handle_t nullProviderCreate(void) {
    struct uma_memory_provider_ops_t ops = {
        .version = UMA_VERSION_CURRENT,
        .initialize = nullInitialize,
        .finalize = nullFinalize,
        .alloc = nullAlloc,
        .free = nullFree,
        .get_last_result = nullGetLastResult,
        .get_recommended_page_size = nullGetRecommendedPageSize,
        .get_min_page_size = nullGetPageSize,
        .purge_lazy = nullPurgeLazy,
        .purge_force = nullPurgeForce,
        .get_name = nullName};

    uma_memory_provider_handle_t hProvider;
    enum uma_result_t ret = umaMemoryProviderCreate(&ops, NULL, &hProvider);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMA_RESULT_SUCCESS);
    return hProvider;
}

struct traceParams {
    uma_memory_provider_handle_t hUpstreamProvider;
    void (*trace)(const char *);
};

static enum uma_result_t traceInitialize(void *params, void **pool) {
    struct traceParams *tracePool =
        (struct traceParams *)malloc(sizeof(struct traceParams));
    *tracePool = *((struct traceParams *)params);
    *pool = tracePool;

    return UMA_RESULT_SUCCESS;
}

static void traceFinalize(void *pool) { free(pool); }

static enum uma_result_t traceAlloc(void *provider, size_t size,
                                    size_t alignment, void **ptr) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("alloc");
    return umaMemoryProviderAlloc(traceProvider->hUpstreamProvider, size,
                                  alignment, ptr);
}

static enum uma_result_t traceFree(void *provider, void *ptr, size_t size) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("free");
    return umaMemoryProviderFree(traceProvider->hUpstreamProvider, ptr, size);
}

static enum uma_result_t traceGetLastResult(void *provider,
                                            const char **ppMsg) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("get_last_result");
    return umaMemoryProviderGetLastResult(traceProvider->hUpstreamProvider,
                                          ppMsg);
}

static enum uma_result_t
traceGetRecommendedPageSize(void *provider, size_t size, size_t *pageSize) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("get_recommended_page_size");
    return umaMemoryProviderGetRecommendedPageSize(
        traceProvider->hUpstreamProvider, size, pageSize);
}

static enum uma_result_t traceGetPageSize(void *provider, void *ptr,

                                          size_t *pageSize) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("get_min_page_size");
    return umaMemoryProviderGetMinPageSize(traceProvider->hUpstreamProvider,
                                           ptr, pageSize);
}

static enum uma_result_t tracePurgeLazy(void *provider, void *ptr,
                                        size_t size) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("purge_lazy");
    return umaMemoryProviderPurgeLazy(traceProvider->hUpstreamProvider, ptr,
                                      size);
}

static enum uma_result_t tracePurgeForce(void *provider, void *ptr,
                                         size_t size) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("purge_force");
    return umaMemoryProviderPurgeForce(traceProvider->hUpstreamProvider, ptr,
                                       size);
}

static void traceName(void *provider, const char **ppName) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("name");
    umaMemoryProviderGetName(traceProvider->hUpstreamProvider, ppName);
}

uma_memory_provider_handle_t
traceProviderCreate(uma_memory_provider_handle_t hUpstreamProvider,
                    void (*trace)(const char *)) {
    struct uma_memory_provider_ops_t ops = {
        .version = UMA_VERSION_CURRENT,
        .initialize = traceInitialize,
        .finalize = traceFinalize,
        .alloc = traceAlloc,
        .free = traceFree,
        .get_last_result = traceGetLastResult,
        .get_recommended_page_size = traceGetRecommendedPageSize,
        .get_min_page_size = traceGetPageSize,
        .purge_lazy = tracePurgeLazy,
        .purge_force = tracePurgeForce,
        .get_name = traceName};

    struct traceParams params = {.hUpstreamProvider = hUpstreamProvider,
                                 .trace = trace};

    uma_memory_provider_handle_t hProvider;
    enum uma_result_t ret = umaMemoryProviderCreate(&ops, &params, &hProvider);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMA_RESULT_SUCCESS);
    return hProvider;
}
