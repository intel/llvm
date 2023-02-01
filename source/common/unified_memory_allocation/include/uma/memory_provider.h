/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UMA_MEMORY_PROVIDER_H
#define UMA_MEMORY_PROVIDER_H 1

#include <uma/base.h>
#include <uma/memory_provider_ops.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uma_memory_provider_t *uma_memory_provider_handle_t;

///
/// \brief Creates new memory provider
/// \param ops instance of uma_memory_provider_ops_t
/// \param params pointer to provider-specific parameters
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure
///
enum uma_result_t
umaMemoryProviderCreate(struct uma_memory_provider_ops_t *ops, void *params,
                        uma_memory_provider_handle_t *hProvider);

///
/// \brief Destroys memory provider
/// \param hPool handle to the memory provider
///
void umaMemoryProviderDestroy(uma_memory_provider_handle_t hProvider);

///
/// \brief Allocates size bytes of uninitialized storage from memory provider
/// with
///        specified alignment
/// \param hProvider handle to the memory provider
/// \param size number of bytes to allocate
/// \param alignment alignment of the allocation
/// \param ptr returns pointer to the allocated memory
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure
///
enum uma_result_t umaMemoryProviderAlloc(uma_memory_provider_handle_t hProvider,
                                         size_t size, size_t alignment,
                                         void **ptr);

///
/// \brief Frees the memory space pointed by ptr from the memory provider
/// \param hProvider handle to the memory provider
/// \param ptr pointer to the allocated memory
/// \param size size of the allocation
///
enum uma_result_t umaMemoryProviderFree(uma_memory_provider_handle_t hProvider,
                                        void *ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* UMA_MEMORY_PROVIDER_H */
