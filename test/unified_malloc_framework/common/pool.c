// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pool.h"

#include "provider.h"
#include <umf/memory_pool_ops.h>

#include <assert.h>
#include <stdlib.h>

static enum umf_result_t nullInitialize(umf_memory_provider_handle_t *providers,
                                        size_t numProviders, void *params,
                                        void **pool) {
    (void)providers;
    (void)numProviders;
    (void)params;
    assert(providers && numProviders);
    *pool = NULL;
    return UMF_RESULT_SUCCESS;
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

static enum umf_result_t nullFree(void *pool, void *ptr) {
    (void)pool;
    (void)ptr;
    return UMF_RESULT_SUCCESS;
}

enum umf_result_t nullGetLastStatus(void *pool) {
    (void)pool;
    return UMF_RESULT_SUCCESS;
}

umf_memory_pool_handle_t nullPoolCreate(void) {
    struct umf_memory_pool_ops_t ops = {
        .version = UMF_VERSION_CURRENT,
        .initialize = nullInitialize,
        .finalize = nullFinalize,
        .malloc = nullMalloc,
        .realloc = nullRealloc,
        .calloc = nullCalloc,
        .aligned_malloc = nullAlignedMalloc,
        .malloc_usable_size = nullMallocUsableSize,
        .free = nullFree,
        .get_last_allocation_error = nullGetLastStatus};

    umf_memory_provider_handle_t providerDesc = nullProviderCreate();
    umf_memory_pool_handle_t hPool;
    enum umf_result_t ret = umfPoolCreate(&ops, &providerDesc, 1, NULL, &hPool);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMF_RESULT_SUCCESS);
    return hPool;
}

struct traceParams {
    umf_memory_pool_handle_t hUpstreamPool;
    void (*trace)(const char *);
};

struct tracePool {
    struct traceParams params;
};

static enum umf_result_t
traceInitialize(umf_memory_provider_handle_t *providers, size_t numProviders,
                void *params, void **pool) {
    struct tracePool *tracePool =
        (struct tracePool *)malloc(sizeof(struct tracePool));
    tracePool->params = *((struct traceParams *)params);

    (void)providers;
    (void)numProviders;
    assert(providers && numProviders);

    *pool = tracePool;
    return UMF_RESULT_SUCCESS;
}

static void traceFinalize(void *pool) { free(pool); }

static void *traceMalloc(void *pool, size_t size) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("malloc");
    return umfPoolMalloc(tracePool->params.hUpstreamPool, size);
}

static void *traceCalloc(void *pool, size_t num, size_t size) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("calloc");
    return umfPoolCalloc(tracePool->params.hUpstreamPool, num, size);
}

static void *traceRealloc(void *pool, void *ptr, size_t size) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("realloc");
    return umfPoolRealloc(tracePool->params.hUpstreamPool, ptr, size);
}

static void *traceAlignedMalloc(void *pool, size_t size, size_t alignment) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("aligned_malloc");
    return umfPoolAlignedMalloc(tracePool->params.hUpstreamPool, size,
                                alignment);
}

static size_t traceMallocUsableSize(void *pool, void *ptr) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("malloc_usable_size");
    return umfPoolMallocUsableSize(tracePool->params.hUpstreamPool, ptr);
}

static enum umf_result_t traceFree(void *pool, void *ptr) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("free");
    return umfPoolFree(tracePool->params.hUpstreamPool, ptr);
}

enum umf_result_t traceGetLastStatus(void *pool) {
    struct tracePool *tracePool = (struct tracePool *)pool;

    tracePool->params.trace("get_last_native_error");
    return umfPoolGetLastAllocationError(tracePool->params.hUpstreamPool);
}

umf_memory_pool_handle_t
tracePoolCreate(umf_memory_pool_handle_t hUpstreamPool,
                umf_memory_provider_handle_t providerDesc,
                void (*trace)(const char *)) {
    struct umf_memory_pool_ops_t ops = {
        .version = UMF_VERSION_CURRENT,
        .initialize = traceInitialize,
        .finalize = traceFinalize,
        .malloc = traceMalloc,
        .realloc = traceRealloc,
        .calloc = traceCalloc,
        .aligned_malloc = traceAlignedMalloc,
        .malloc_usable_size = traceMallocUsableSize,
        .free = traceFree,
        .get_last_allocation_error = traceGetLastStatus};

    struct traceParams params = {.hUpstreamPool = hUpstreamPool,
                                 .trace = trace};

    umf_memory_pool_handle_t hPool;
    enum umf_result_t ret =
        umfPoolCreate(&ops, &providerDesc, 1, &params, &hPool);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMF_RESULT_SUCCESS);
    return hPool;
}
