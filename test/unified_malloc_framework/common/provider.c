// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider.h"

#include <umf/memory_provider_ops.h>

#include <assert.h>
#include <stdlib.h>

static enum umf_result_t nullInitialize(void *params, void **pool) {
    (void)params;
    *pool = NULL;
    return UMF_RESULT_SUCCESS;
}

static void nullFinalize(void *pool) { (void)pool; }

static enum umf_result_t nullAlloc(void *provider, size_t size,
                                   size_t alignment, void **ptr) {
    (void)provider;
    (void)size;
    (void)alignment;
    *ptr = NULL;
    return UMF_RESULT_SUCCESS;
}

static enum umf_result_t nullFree(void *provider, void *ptr, size_t size) {
    (void)provider;
    (void)ptr;
    (void)size;
    return UMF_RESULT_SUCCESS;
}

static void nullGetLastError(void *provider, const char **ppMsg,
                             int32_t *pError) {
    (void)provider;
    (void)ppMsg;
    (void)pError;
}

static enum umf_result_t nullGetRecommendedPageSize(void *provider, size_t size,
                                                    size_t *pageSize) {
    (void)provider;
    (void)size;
    (void)pageSize;
    return UMF_RESULT_SUCCESS;
}

static enum umf_result_t nullGetPageSize(void *provider, void *ptr,

                                         size_t *pageSize) {
    (void)provider;
    (void)ptr;
    (void)pageSize;
    return UMF_RESULT_SUCCESS;
}

static enum umf_result_t nullPurgeLazy(void *provider, void *ptr, size_t size) {
    (void)provider;
    (void)ptr;
    (void)size;
    return UMF_RESULT_SUCCESS;
}

static enum umf_result_t nullPurgeForce(void *provider, void *ptr,
                                        size_t size) {
    (void)provider;
    (void)ptr;
    (void)size;
    return UMF_RESULT_SUCCESS;
}

static const char *nullName(void *provider) {
    (void)provider;
    return "null";
}

umf_memory_provider_handle_t nullProviderCreate(void) {
    struct umf_memory_provider_ops_t ops = {
        .version = UMF_VERSION_CURRENT,
        .initialize = nullInitialize,
        .finalize = nullFinalize,
        .alloc = nullAlloc,
        .free = nullFree,
        .get_last_native_error = nullGetLastError,
        .get_recommended_page_size = nullGetRecommendedPageSize,
        .get_min_page_size = nullGetPageSize,
        .purge_lazy = nullPurgeLazy,
        .purge_force = nullPurgeForce,
        .get_name = nullName};

    umf_memory_provider_handle_t hProvider;
    enum umf_result_t ret = umfMemoryProviderCreate(&ops, NULL, &hProvider);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMF_RESULT_SUCCESS);
    return hProvider;
}

struct traceParams {
    umf_memory_provider_handle_t hUpstreamProvider;
    void (*trace)(const char *);
};

static enum umf_result_t traceInitialize(void *params, void **pool) {
    struct traceParams *tracePool =
        (struct traceParams *)malloc(sizeof(struct traceParams));
    *tracePool = *((struct traceParams *)params);
    *pool = tracePool;

    return UMF_RESULT_SUCCESS;
}

static void traceFinalize(void *pool) { free(pool); }

static enum umf_result_t traceAlloc(void *provider, size_t size,
                                    size_t alignment, void **ptr) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("alloc");
    return umfMemoryProviderAlloc(traceProvider->hUpstreamProvider, size,
                                  alignment, ptr);
}

static enum umf_result_t traceFree(void *provider, void *ptr, size_t size) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("free");
    return umfMemoryProviderFree(traceProvider->hUpstreamProvider, ptr, size);
}

static void traceGetLastError(void *provider, const char **ppMsg,
                              int32_t *pError) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("get_last_native_error");
    umfMemoryProviderGetLastNativeError(traceProvider->hUpstreamProvider, ppMsg,
                                        pError);
}

static enum umf_result_t
traceGetRecommendedPageSize(void *provider, size_t size, size_t *pageSize) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("get_recommended_page_size");
    return umfMemoryProviderGetRecommendedPageSize(
        traceProvider->hUpstreamProvider, size, pageSize);
}

static enum umf_result_t traceGetPageSize(void *provider, void *ptr,

                                          size_t *pageSize) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("get_min_page_size");
    return umfMemoryProviderGetMinPageSize(traceProvider->hUpstreamProvider,
                                           ptr, pageSize);
}

static enum umf_result_t tracePurgeLazy(void *provider, void *ptr,
                                        size_t size) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("purge_lazy");
    return umfMemoryProviderPurgeLazy(traceProvider->hUpstreamProvider, ptr,
                                      size);
}

static enum umf_result_t tracePurgeForce(void *provider, void *ptr,
                                         size_t size) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("purge_force");
    return umfMemoryProviderPurgeForce(traceProvider->hUpstreamProvider, ptr,
                                       size);
}

static const char *traceName(void *provider) {
    struct traceParams *traceProvider = (struct traceParams *)provider;

    traceProvider->trace("name");
    return umfMemoryProviderGetName(traceProvider->hUpstreamProvider);
}

umf_memory_provider_handle_t
traceProviderCreate(umf_memory_provider_handle_t hUpstreamProvider,
                    void (*trace)(const char *)) {
    struct umf_memory_provider_ops_t ops = {
        .version = UMF_VERSION_CURRENT,
        .initialize = traceInitialize,
        .finalize = traceFinalize,
        .alloc = traceAlloc,
        .free = traceFree,
        .get_last_native_error = traceGetLastError,
        .get_recommended_page_size = traceGetRecommendedPageSize,
        .get_min_page_size = traceGetPageSize,
        .purge_lazy = tracePurgeLazy,
        .purge_force = tracePurgeForce,
        .get_name = traceName};

    struct traceParams params = {.hUpstreamProvider = hUpstreamProvider,
                                 .trace = trace};

    umf_memory_provider_handle_t hProvider;
    enum umf_result_t ret = umfMemoryProviderCreate(&ops, &params, &hProvider);

    (void)ret; /* silence unused variable warning */
    assert(ret == UMF_RESULT_SUCCESS);
    return hProvider;
}
