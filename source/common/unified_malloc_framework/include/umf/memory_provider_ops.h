/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_PROVIDER_OPS_H
#define UMF_MEMORY_PROVIDER_OPS_H 1

#include <umf/base.h>

#ifdef __cplusplus
extern "C" {
#endif

/// This structure comprises function pointers used by corresponding
/// umfMemoryProvider* calls. Each memory provider implementation should
/// initialize all function pointers.
struct umf_memory_provider_ops_t {
    /// Version of the ops structure.
    /// Should be initialized using UMF_VERSION_CURRENT
    uint32_t version;

    ///
    /// \brief Initializes memory provider.
    /// \param params provider-specific params
    /// \param provider returns pointer to the provider
    /// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
    enum umf_result_t (*initialize)(void *params, void **provider);

    ///
    /// \brief Finalizes memory provider.
    /// \param provider provider to finalize
    void (*finalize)(void *provider);

    /// Refer to memory_provider.h for description of those functions
    enum umf_result_t (*alloc)(void *provider, size_t size, size_t alignment,
                               void **ptr);
    enum umf_result_t (*free)(void *provider, void *ptr, size_t size);
    void (*get_last_native_error)(void *provider, const char **ppMessage,
                                  int32_t *pError);
    enum umf_result_t (*get_recommended_page_size)(void *provider, size_t size,
                                                   size_t *pageSize);
    enum umf_result_t (*get_min_page_size)(void *provider, void *ptr,
                                           size_t *pageSize);
    enum umf_result_t (*purge_lazy)(void *provider, void *ptr, size_t size);
    enum umf_result_t (*purge_force)(void *provider, void *ptr, size_t size);
    const char *(*get_name)(void *provider);
};

#ifdef __cplusplus
}
#endif

#endif /* #ifndef UMF_MEMORY_PROVIDER_OPS_H */
