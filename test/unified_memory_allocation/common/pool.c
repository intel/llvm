// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "pool.h"

#include "provider.h"
#include <uma/memory_pool_ops.h>

#include <assert.h>
#include <stdlib.h>

static enum uma_result_t nullInitialize(uma_memory_provider_handle_t *providers,
                                        size_t numProviders, void *params,
                                        void **pool) {
    (void)providers;
    (void)numProviders;
    (void)params;
    assert(providers && numProviders);
    *pool = NULL;
    return UMA_RESULT_SUCCESS;
}

static void nullFinalize(void *pool) { (void)pool; }

static void *nullMalloc(void *pool, size_t size) {
    (void)pool;
    (void)size;
    return NULL;
}

static void *nullCalloc(void *pool, size_t num, size_t size) {
    (void)pool;
    (void)num;
    (void)size;
    return NULL;
}

static void *nullRealloc(void *pool, void *ptr, size_t size) {
    (void)pool;
    (void)ptr;
    (void)size;
    return NULL;
}

static void *nullAlignedMalloc(void *pool, size_t size, size_t alignment) {
    (void)pool;
    (void)size;
    (void)alignment;
    return NULL;
}

static size_t nullMallocUsableSize(void *pool, void *ptr) {
    (void)ptr;
    (void)pool;
    return 0;
}

static void nullFree(void *pool, void *ptr) {
    (void)pool;
    (void)ptr;
}

enum uma_result_t nullGetLastResult(void *pool, const char **ppMsg) {
    (void)pool;
    (void)ppMsg;
    return UMA_RESULT_SUCCESS;
}

uma_memory_pool_handle_t nullPoolCreate(void) {
    struct uma_memory_pool_ops_t ops = {.version = UMA_VERSION_CURRENT,
                                        .initialize = nullInitialize,
                                        .finalize = nullFinalize,
                                        .malloc = nullMalloc,
                                        .realloc = nullRealloc,
                                        .calloc = nullCalloc,
                                        .aligned_malloc = nullAlignedMalloc,
                                        .malloc_usable_size =
                                            nullMallocUsableSize,
                                        .free = nullFree,
                                        .get_last_result = nullGetLastResult};

    uma_memory_provider_handle_t providerDesc = nullProviderCreate();
    uma_memory_pool_handle_t hPool;
    enum uma_result_t ret = umaPoolCreate(&ops, &providerDesc, 1, NULL, &hPool);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMA_RESULT_SUCCESS);
    return hPool;
}

struct traceParams {
    uma_memory_pool_handle_t hUpstreamPool;
    void (*trace)(const char *);
};

struct tracePool {
    struct traceParams params;
};

static enum uma_result_t
traceInitialize(uma_memory_provider_handle_t *providers, size_t numProviders,
                void *params, void **pool) {
    struct tracePool *tracePool =
        (struct tracePool *)malloc(sizeof(struct tracePool));
    tracePool->params = *((struct traceParams *)params);

    (void)providers;
    (void)numProviders;
    assert(providers && numProviders);

    *pool = tracePool;
    return UMA_RESULT_SUCCESS;
}

static void traceFinalize(void *pool) { free(pool); }

static void *traceMalloc(void *pool, size_t size) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("malloc");
    return umaPoolMalloc(tracePool->params.hUpstreamPool, size);
}

static void *traceCalloc(void *pool, size_t num, size_t size) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("calloc");
    return umaPoolCalloc(tracePool->params.hUpstreamPool, num, size);
}

static void *traceRealloc(void *pool, void *ptr, size_t size) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("realloc");
    return umaPoolRealloc(tracePool->params.hUpstreamPool, ptr, size);
}

static void *traceAlignedMalloc(void *pool, size_t size, size_t alignment) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("aligned_malloc");
    return umaPoolAlignedMalloc(tracePool->params.hUpstreamPool, size,
                                alignment);
}

static size_t traceMallocUsableSize(void *pool, void *ptr) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("malloc_usable_size");
    return umaPoolMallocUsableSize(tracePool->params.hUpstreamPool, ptr);
}

static void traceFree(void *pool, void *ptr) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("free");
    umaPoolFree(tracePool->params.hUpstreamPool, ptr);
}

enum uma_result_t traceGetLastResult(void *pool, const char **ppMsg) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("get_last_result");
    return umaPoolGetLastResult(tracePool->params.hUpstreamPool, ppMsg);
}

uma_memory_pool_handle_t
tracePoolCreate(uma_memory_pool_handle_t hUpstreamPool,
                uma_memory_provider_handle_t providerDesc,
                void (*trace)(const char *)) {
    struct uma_memory_pool_ops_t ops = {.version = UMA_VERSION_CURRENT,
                                        .initialize = traceInitialize,
                                        .finalize = traceFinalize,
                                        .malloc = traceMalloc,
                                        .realloc = traceRealloc,
                                        .calloc = traceCalloc,
                                        .aligned_malloc = traceAlignedMalloc,
                                        .malloc_usable_size =
                                            traceMallocUsableSize,
                                        .free = traceFree,
                                        .get_last_result = traceGetLastResult};

    struct traceParams params = {.hUpstreamPool = hUpstreamPool,
                                 .trace = trace};

    uma_memory_pool_handle_t hPool;
    enum uma_result_t ret =
        umaPoolCreate(&ops, &providerDesc, 1, &params, &hPool);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMA_RESULT_SUCCESS);
    return hPool;
}
