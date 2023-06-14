/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMA_MEMORY_PROVIDER_OPS_H
#define UMA_MEMORY_PROVIDER_OPS_H 1

#include <uma/base.h>

#ifdef __cplusplus
extern "C" {
#endif

/// This structure comprises function pointers used by corresponding
/// umaMemoryProvider* calls. Each memory provider implementation should
/// initialize all function pointers.
struct uma_memory_provider_ops_t {
    /// Version of the ops structure.
    /// Should be initialized using UMA_VERSION_CURRENT
    uint32_t version;

    ///
    /// \brief Intializes memory pool.
    /// \param params pool-specific params
    /// \param pool returns pointer to the pool
    /// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
    enum uma_result_t (*initialize)(void *params, void **pool);

    ///
    /// \brief Finalizes memory pool.
    /// \param pool pool to finalize
    void (*finalize)(void *pool);

    /// Refer to memory_provider.h for description of those functions
    enum uma_result_t (*alloc)(void *provider, size_t size, size_t alignment,
                               void **ptr);
    enum uma_result_t (*free)(void *provider, void *ptr, size_t size);
    enum uma_result_t (*get_last_result)(void *provider,
                                         const char **ppMessage);
    enum uma_result_t (*get_recommended_page_size)(void *provider, size_t size,
                                                   size_t *pageSize);
    enum uma_result_t (*get_min_page_size)(void *provider, void *ptr,
                                           size_t *pageSize);
    enum uma_result_t (*purge_lazy)(void *provider, void *ptr, size_t size);
    enum uma_result_t (*purge_force)(void *provider, void *ptr, size_t size);
    void (*get_name)(void *provider, const char **ppName);
};

#ifdef __cplusplus
}
#endif

#endif /* #ifndef UMA_MEMORY_PROVIDER_OPS_H */
