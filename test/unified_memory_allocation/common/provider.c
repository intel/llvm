// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "provider.h"

#include <uma/memory_provider_ops.h>

#include <assert.h>
#include <stdlib.h>

enum uma_result_t nullInitialize(void *params, void **pool) {
    (void) params;
    *pool = NULL;
    return UMA_RESULT_SUCCESS;
}

void nullFinalize(void *pool) {
    (void) pool;
}

enum uma_result_t nullAlloc(void *provider, size_t size, size_t alignment, void **ptr) {
    (void) provider;
    (void) size;
    (void) alignment;
    (void) ptr;
    return UMA_RESULT_SUCCESS;
}

enum uma_result_t nullFree(void *provider, void *ptr, size_t size) {
    (void) provider;
    (void) ptr;
    (void) size;
    return UMA_RESULT_SUCCESS;
}

uma_memory_provider_handle_t nullProviderCreate() {
    struct uma_memory_provider_ops_t ops = {
        .version = UMA_VERSION_CURRENT,
        .initialize = nullInitialize,
        .finalize = nullFinalize,
        .alloc = nullAlloc,
        .free = nullFree
    };

    uma_memory_provider_handle_t hProvider;
    enum uma_result_t ret = umaMemoryProviderCreate(&ops, NULL,
                                     &hProvider);
    
    assert(ret == UMA_RESULT_SUCCESS);
    return hProvider;
}

struct traceParams {
    uma_memory_provider_handle_t hUpstreamProvider;
    void (*trace)(const char*);
};

enum uma_result_t traceInitialize(void *params, void **pool) {
    struct traceParams* tracePool = (struct traceParams*) malloc(sizeof(struct traceParams));
    *tracePool = *((struct traceParams*) params);
    *pool = tracePool;

    return UMA_RESULT_SUCCESS;
}

void traceFinalize(void *pool) {
    free(pool);
}

enum uma_result_t traceAlloc(void *provider, size_t size, size_t alignment, void **ptr) {
    struct traceParams* traceProvider = (struct traceParams*) provider;

    traceProvider->trace("alloc");
    return umaMemoryProviderAlloc(traceProvider->hUpstreamProvider, size, alignment, ptr);
}

enum uma_result_t traceFree(void *provider, void *ptr, size_t size) {
    struct traceParams* traceProvider = (struct traceParams*) provider;

    traceProvider->trace("free");
    return umaMemoryProviderFree(traceProvider->hUpstreamProvider, ptr, size);
}

uma_memory_provider_handle_t traceProviderCreate(uma_memory_provider_handle_t hUpstreamProvider,  void (*trace)(const char*)) {
    struct uma_memory_provider_ops_t ops = {
        .version = UMA_VERSION_CURRENT,
        .initialize = traceInitialize,
        .finalize = traceFinalize,
        .alloc = traceAlloc,
        .free = traceFree
    };

    struct traceParams params = {
        .hUpstreamProvider = hUpstreamProvider,
        .trace = trace
    };

    uma_memory_provider_handle_t hProvider;
    enum uma_result_t ret = umaMemoryProviderCreate(&ops, &params,
                                     &hProvider);
    
    assert(ret == UMA_RESULT_SUCCESS);
    return hProvider;
}
