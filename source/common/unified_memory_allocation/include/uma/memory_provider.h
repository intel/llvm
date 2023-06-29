/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
/// \brief Creates new memory provider.
/// \param ops instance of uma_memory_provider_ops_t
/// \param params pointer to provider-specific parameters
/// \param hProvider [out] pointer to the newly created memory provider
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
///
enum uma_result_t
umaMemoryProviderCreate(struct uma_memory_provider_ops_t *ops, void *params,
                        uma_memory_provider_handle_t *hProvider);

///
/// \brief Destroys memory provider.
/// \param hPool handle to the memory provider
///
void umaMemoryProviderDestroy(uma_memory_provider_handle_t hProvider);

///
/// \brief Allocates size bytes of uninitialized storage from memory provider
///        with specified alignment.
/// \param hProvider handle to the memory provider
/// \param size number of bytes to allocate
/// \param alignment alignment of the allocation
/// \param ptr [out] pointer to the allocated memory
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
///
enum uma_result_t umaMemoryProviderAlloc(uma_memory_provider_handle_t hProvider,
                                         size_t size, size_t alignment,
                                         void **ptr);

///
/// \brief Frees the memory space pointed by ptr from the memory provider.
/// \param hProvider handle to the memory provider
/// \param ptr pointer to the allocated memory
/// \param size size of the allocation
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
///
enum uma_result_t umaMemoryProviderFree(uma_memory_provider_handle_t hProvider,
                                        void *ptr, size_t size);

///
/// \brief Retrieve string representation of the underlying provider specific
///        result reported by the last API that returned
///        UMA_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC. Allows for a provider
///        independent way to return a provider specific result.
///
/// \details
///     - The string returned via the ppMessage is a NULL terminated C style
///       string.
///     - The string returned via the ppMessage is thread local.
///     - The memory in the string returned via the ppMessage is owned by the
///       adapter.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// \param hProvider handle to the memory provider
/// \param ppMessage [out] pointer to a string containing provider specific
///        result in string representation
/// \return UMA_RESULT_SUCCESS if the result being reported is to be considered
///         a warning. Any other result code returned indicates that the
///         adapter specific result is an error.
enum uma_result_t
umaMemoryProviderGetLastResult(uma_memory_provider_handle_t hProvider,
                               const char **ppMessage);

///
/// \brief Retrieve recommended page size for a given allocation size.
/// \param hProvider handle to the memory provider
/// \param size allocation size
/// \param pageSize [out] will be updated with recommended page size
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
enum uma_result_t
umaMemoryProviderGetRecommendedPageSize(uma_memory_provider_handle_t hProvider,
                                        size_t size, size_t *pageSize);

///
/// \brief Retrieve minimum possible page size used by memory region referenced by given ptr
///        or minimum possible page size that can be used by this provider if ptr is NULL.
/// \param hProvider handle to the memory provider
/// \param ptr [optional] pointer to memory allocated by this memory provider
/// \param pageSize [out] will be updated with page size value.
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
enum uma_result_t
umaMemoryProviderGetMinPageSize(uma_memory_provider_handle_t hProvider,
                                void *ptr, size_t *pageSize);

///
/// \brief Discard physical pages within the virtual memory mapping associated at given addr and size.
///        This call is asynchronous and may delay puring the pages indefinitely.
/// \param hProvider handle to the memory provider
/// \param ptr beginning of the virtual memory range
/// \param size size of the virtual memory range
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMA_RESULT_ERROR_INVALID_ALIGNMENT if ptr or size is not page-aligned.
///         UMA_RESULT_ERROR_NOT_SUPPORTED if operation is not supported by this provider.
enum uma_result_t
umaMemoryProviderPurgeLazy(uma_memory_provider_handle_t hProvider, void *ptr,
                           size_t size);

///
/// \brief Discard physical pages within the virtual memory mapping associated at given addr and size.
///        This call is synchronous and if it suceeds, pages are guaranteed to be zero-filled on the next access.
/// \param hProvider handle to the memory provider
/// \param ptr beginning of the virtual memory range
/// \param size size of the virtual memory range
/// \return UMA_RESULT_SUCCESS on success or appropriate error code on failure
///         UMA_RESULT_ERROR_INVALID_ALIGNMENT if ptr or size is not page-aligned.
///         UMA_RESULT_ERROR_NOT_SUPPORTED if operation is not supported by this provider.
enum uma_result_t
umaMemoryProviderPurgeForce(uma_memory_provider_handle_t hProvider, void *ptr,
                            size_t size);

///
/// \brief Retrive name of a given memory provider.
/// \param hProvider handle to the memory provider
/// \param ppName [out] pointer to a string containing name of the provider
void umaMemoryProviderGetName(uma_memory_provider_handle_t hProvider,
                              const char **ppName);

#ifdef __cplusplus
}
#endif

#endif /* UMA_MEMORY_PROVIDER_H */
