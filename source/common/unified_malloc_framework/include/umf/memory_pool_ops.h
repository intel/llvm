/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_POOL_OPS_H
#define UMF_MEMORY_POOL_OPS_H 1

#include <umf/base.h>
#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief This structure comprises function pointers used by corresponding umfPool*
/// calls. Each memory pool implementation should initialize all function
/// pointers.
struct umf_memory_pool_ops_t {
    /// Version of the ops structure.
    /// Should be initialized using UMF_VERSION_CURRENT
    uint32_t version;

    ///
    /// \brief Initializes memory pool.
    /// \param providers array of memory providers that will be used for coarse-grain allocations.
    ///        Should contain at least one memory provider.
    /// \param numProvider number of elements in the providers array
    /// \param params pool-specific params
    /// \param pool [out] returns pointer to the pool
    /// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
    enum umf_result_t (*initialize)(umf_memory_provider_handle_t *providers,
                                    size_t numProviders, void *params,
                                    void **pool);

    ///
    /// \brief Finalizes memory pool
    /// \param pool pool to finalize
    void (*finalize)(void *pool);

    /// Refer to memory_pool.h for description of those functions
    void *(*malloc)(void *pool, size_t size);
    void *(*calloc)(void *pool, size_t num, size_t size);
    void *(*realloc)(void *pool, void *ptr, size_t size);
    void *(*aligned_malloc)(void *pool, size_t size, size_t alignment);
    size_t (*malloc_usable_size)(void *pool, void *ptr);
    enum umf_result_t (*free)(void *pool, void *);
    enum umf_result_t (*get_last_allocation_error)(void *pool);
};

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_POOL_OPS_H */
