// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "pool.h"

#include <uma/memory_pool_ops.h>

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

void *nullMalloc(void *pool, size_t size) {
    (void) pool;
    (void) size;
    return NULL;
}

void *nullCalloc(void *pool, size_t num, size_t size) {
    (void) pool;
    (void) num;
    (void) size;
    return NULL;
}

void *nullRealloc(void *pool, void *ptr, size_t size) {
    (void) pool;
    (void) ptr;
    (void) size;
    return NULL;
}

void *nullAlignedMalloc(void *pool, size_t size, size_t alignment) {
    (void) pool;
    (void) size;
    (void) alignment;
    return NULL;
}

size_t nullMallocUsableSize(void *pool, void *ptr) {
    (void) ptr;
    (void) pool;
    return 0;
}

void nullFree(void *pool, void *ptr) {
    (void) pool;
    (void) ptr;
}

uma_memory_pool_handle_t nullPoolCreate() {
    struct uma_memory_pool_ops_t ops = {
        .version = UMA_VERSION_CURRENT,
        .initialize = nullInitialize,
        .finalize = nullFinalize,
        .malloc = nullMalloc,
        .realloc = nullRealloc,
        .calloc = nullCalloc,
        .aligned_malloc = nullAlignedMalloc,
        .malloc_usable_size = nullMallocUsableSize,
        .free = nullFree
    };

    uma_memory_pool_handle_t hPool;
    enum uma_result_t ret = umaPoolCreate(&ops, NULL,
                                     &hPool);
    
    assert(ret == UMA_RESULT_SUCCESS);
    return hPool;
}

struct traceParams {
    uma_memory_pool_handle_t hUpstreamPool;
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

void *traceMalloc(void *pool, size_t size) {
    struct traceParams* tracePool = (struct traceParams*) pool;

    tracePool->trace("malloc");
    return umaPoolMalloc(tracePool->hUpstreamPool, size);
}

void *traceCalloc(void *pool, size_t num, size_t size) {
    struct traceParams* tracePool = (struct traceParams*) pool;

    tracePool->trace("calloc");
    return umaPoolCalloc(tracePool->hUpstreamPool, num, size);
}

void *traceRealloc(void *pool, void *ptr, size_t size) {
    struct traceParams* tracePool = (struct traceParams*) pool;
    
    tracePool->trace("realloc");
    return umaPoolRealloc(tracePool->hUpstreamPool, ptr, size);
}

void *traceAlignedMalloc(void *pool, size_t size, size_t alignment) {
    struct traceParams* tracePool = (struct traceParams*) pool;

    tracePool->trace("aligned_malloc");
    return umaPoolAlignedMalloc(tracePool->hUpstreamPool, size, alignment);
}

size_t traceMallocUsableSize(void *pool, void *ptr) {
    struct traceParams* tracePool = (struct traceParams*) pool;

    tracePool->trace("malloc_usable_size");
    return umaPoolMallocUsableSize(tracePool->hUpstreamPool, ptr);
}

void traceFree(void *pool, void *ptr) {
    struct traceParams* tracePool = (struct traceParams*) pool;

    tracePool->trace("free");
    return umaPoolFree(tracePool->hUpstreamPool, ptr);
}

uma_memory_pool_handle_t tracePoolCreate(uma_memory_pool_handle_t hUpstreamPool,  void (*trace)(const char*)) {
    struct uma_memory_pool_ops_t ops = {
        .version = UMA_VERSION_CURRENT,
        .initialize = traceInitialize,
        .finalize = traceFinalize,
        .malloc = traceMalloc,
        .realloc = traceRealloc,
        .calloc = traceCalloc,
        .aligned_malloc = traceAlignedMalloc,
        .malloc_usable_size = traceMallocUsableSize,
        .free = traceFree
    };

    struct traceParams params = {
        .hUpstreamPool = hUpstreamPool,
        .trace = trace
    };

    uma_memory_pool_handle_t hPool;
    enum uma_result_t ret = umaPoolCreate(&ops, &params,
                                     &hPool);
    
    assert(ret == UMA_RESULT_SUCCESS);
    return hPool;
}
