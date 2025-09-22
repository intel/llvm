//==----------- async_alloc.hpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/detail/common.hpp> // for code_location
#include <sycl/handler.hpp>       // for handler
#include <sycl/queue.hpp>         // for queue
#include <sycl/usm/usm_enums.hpp> // for usm::alloc

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Forward declare memory_pool.
class memory_pool;

/**
 * @brief  Asynchronousy allocate memory from a default pool.
 *
 * @param  q The queue with which to enqueue the asynchronous allocation.
 * @param  kind The kind of memory pool allocation - device, host, shared, etc.
 * @param  size The size in bytes to allocate.
 *
 * @return Generic pointer to allocated USM memory.
 */
__SYCL_EXPORT void *async_malloc(const sycl::queue &q, sycl::usm::alloc kind,
                                 size_t size,
                                 const sycl::detail::code_location &CodeLoc =
                                     sycl::detail::code_location::current());

/**
 * @brief  Asynchronously allocate memory from a default pool.
 *
 * @param  h The handler with which to enqueue the asynchronous allocation.
 * @param  kind The kind of memory pool allocation - device, host, shared, etc.
 * @param  size The size in bytes to allocate.
 *
 * @return Generic pointer to allocated USM memory.
 */
__SYCL_EXPORT void *async_malloc(sycl::handler &h, sycl::usm::alloc kind,
                                 size_t size);

/**
 * @brief  Asynchronously allocate memory from a specified pool.
 *
 * @param  q The queue with which to enqueue the asynchronous allocation.
 * @param  size The size in bytes to allocate.
 * @param  pool The pool with which to allocate from.
 *
 * @return Generic pointer to allocated USM memory.
 */
__SYCL_EXPORT void *
async_malloc_from_pool(const sycl::queue &q, size_t size,
                       const memory_pool &pool,
                       const sycl::detail::code_location &CodeLoc =
                           sycl::detail::code_location::current());

/**
 * @brief  Asynchronously allocate memory from a specified pool.
 *
 * @param  h The handler with which to enqueue the asynchronous allocation.
 * @param  size The size in bytes to allocate.
 * @param  pool The pool with which to allocate from.
 *
 * @return Generic pointer to allocated USM memory.
 */
__SYCL_EXPORT void *async_malloc_from_pool(sycl::handler &h, size_t size,
                                           const memory_pool &pool);

/**
 * @brief  Asynchronously free memory.
 *
 * @param  q The queue with which to enqueue the asynchronous free.
 * @param  ptr The generic pointer to be freed.
 */
__SYCL_EXPORT void async_free(const sycl::queue &q, void *ptr,
                              const sycl::detail::code_location &CodeLoc =
                                  sycl::detail::code_location::current());

/**
 * @brief  Asynchronously free memory.
 *
 * @param  h The handler with which to enqueue the asynchronous free.
 * @param  ptr The generic pointer to be freed.
 */
__SYCL_EXPORT void async_free(sycl::handler &h, void *ptr);

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
