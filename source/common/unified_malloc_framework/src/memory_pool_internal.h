/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_POOL_INTERNAL_H
#define UMF_MEMORY_POOL_INTERNAL_H 1

#include <umf/base.h>
#include <umf/memory_pool_ops.h>
#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

struct umf_memory_pool_t {
    void *pool_priv;
    struct umf_memory_pool_ops_t ops;

    // Holds array of memory providers. All providers are wrapped
    // by memory tracking providers (owned and released by UMF).
    umf_memory_provider_handle_t *providers;

    size_t numProviders;
};

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_POOL_INTERNAL_H */
