/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_POOL_H
#define UMF_MEMORY_POOL_H 1

#include <umf/base.h>
#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct umf_memory_pool_t *umf_memory_pool_handle_t;

struct umf_memory_pool_ops_t;

///
/// \brief Creates new memory pool.
/// \param ops instance of umf_memory_pool_ops_t
/// \param providers array of memory providers that will be used for coarse-grain allocations.
///        Should contain at least one memory provider.
/// \param numProvider number of elements in the providers array
/// \param params pointer to pool-specific parameters
/// \param hPool [out] handle to the newly created memory pool
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
enum umf_result_t umfPoolCreate(const struct umf_memory_pool_ops_t *ops,
                                umf_memory_provider_handle_t *providers,
                                size_t numProviders, void *params,
                                umf_memory_pool_handle_t *hPool);

///
/// \brief Destroys memory pool.
/// \param hPool handle to the pool
///
void umfPoolDestroy(umf_memory_pool_handle_t hPool);

///
/// \brief Allocates size bytes of uninitialized storage of the specified hPool.
/// \param hPool specified memory hPool
/// \param size number of bytes to allocate
/// \return Pointer to the allocated memory.
///
void *umfPoolMalloc(umf_memory_pool_handle_t hPool, size_t size);

///
/// \brief Allocates size bytes of uninitialized storage of the specified hPool.
/// with specified alignment
/// \param hPool specified memory hPool
/// \param size number of bytes to allocate
/// \param alignment alignment of the allocation
/// \return Pointer to the allocated memory.
///
void *umfPoolAlignedMalloc(umf_memory_pool_handle_t hPool, size_t size,
                           size_t alignment);

///
/// \brief Allocates memory of the specified hPool for an array of num elements
///        of size bytes each and initializes all bytes in the allocated storage
///        to zero.
/// \param hPool specified memory hPool
/// \param num number of objects
/// \param size specified size of each element
/// \return Pointer to the allocated memory.
///
void *umfPoolCalloc(umf_memory_pool_handle_t hPool, size_t num, size_t size);

///
/// \brief Reallocates memory of the specified hPool.
/// \param hPool specified memory hPool
/// \param ptr pointer to the memory block to be reallocated
/// \param size new size for the memory block in bytes
/// \return Pointer to the allocated memory.
///
void *umfPoolRealloc(umf_memory_pool_handle_t hPool, void *ptr, size_t size);

///
/// \brief Obtains size of block of memory allocated from the pool.
/// \param hPool specified memory hPool
/// \param ptr pointer to the allocated memory
/// \return Number of bytes.
///
size_t umfPoolMallocUsableSize(umf_memory_pool_handle_t hPool, void *ptr);

///
/// \brief Frees the memory space of the specified hPool pointed by ptr.
/// \param hPool specified memory hPool
/// \param ptr pointer to the allocated memory
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         Whether any status other than UMF_RESULT_SUCCESS can be returned
///         depends on the memory provider used by the pool.
///
enum umf_result_t umfPoolFree(umf_memory_pool_handle_t hPool, void *ptr);

///
/// \brief Frees the memory space pointed by ptr if it belongs to UMF pool, does nothing otherwise.
/// \param ptr pointer to the allocated memory
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         Whether any status other than UMF_RESULT_SUCCESS can be returned
///         depends on the memory provider used by the pool.
///
enum umf_result_t umfFree(void *ptr);

///
/// \brief Retrieve umf_result_t representing the error of the last failed allocation
///        operation in this thread (malloc, calloc, realloc, aligned_malloc).
///
/// \details
/// * Implementations *must* store the error code in thread-local
///   storage prior to returning NULL from the allocation functions.
///
/// * If the last allocation/de-allocation operation succeeded, the value returned by
///   this function is unspecified.
///
/// * The application *may* call this function from simultaneous threads.
///
/// * The implementation of this function *should* be lock-free.
/// \param hPool specified memory hPool
/// \return Error code desciribng the failure of the last failed allocation operation.
///         The value is undefined if the previous allocation was successful.
enum umf_result_t umfPoolGetLastAllocationError(umf_memory_pool_handle_t hPool);

///
/// \brief Retrieve memory pool associated with a given ptr. Only memory allocated
///        with the usage of a memory provider is being tracked.
/// \param ptr pointer to memory belonging to a memory pool
/// \return Handle to a memory pool that contains ptr or NULL if pointer does not belong to any UMF pool.
umf_memory_pool_handle_t umfPoolByPtr(const void *ptr);

///
/// \brief Retrieve memory providers associated with a given pool.
/// \param hPool specified memory pool
/// \param hProviders [out] pointer to an array of memory providers. If numProviders is not equal to or
///        greater than the real number of providers, UMF_RESULT_ERROR_INVALID_ARGUMENT is returned.
/// \param numProviders [in] number of memory providers to return
/// \param numProvidersRet pointer to the actual number of memory providers
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
enum umf_result_t
umfPoolGetMemoryProviders(umf_memory_pool_handle_t hPool, size_t numProviders,
                          umf_memory_provider_handle_t *hProviders,
                          size_t *numProvidersRet);

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_POOL_H */
