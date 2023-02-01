/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <uma/memory_pool.h>

#include <assert.h>
#include <stdlib.h>

struct uma_memory_pool_t {
  struct uma_memory_pool_ops_t ops;
  void *pool_priv;
};

enum uma_result_t umaPoolCreate(struct uma_memory_pool_ops_t *ops, void *params,
                                uma_memory_pool_handle_t *hPool) {
  uma_memory_pool_handle_t pool = malloc(sizeof(struct uma_memory_pool_t));
  if (!pool) {
    return UMA_RESULT_RUNTIME_ERROR;
  }

  assert(ops->version == UMA_VERSION_CURRENT);

  pool->ops = *ops;

  void *pool_priv;
  enum uma_result_t ret = ops->initialize(params, &pool_priv);
  if (ret != UMA_RESULT_SUCCESS) {
    return ret;
  }

  pool->pool_priv = pool_priv;

  *hPool = pool;

  return UMA_RESULT_SUCCESS;
}

void umaPoolDestroy(uma_memory_pool_handle_t hPool) {
  hPool->ops.finalize(hPool->pool_priv);
  free(hPool);
}

void *umaPoolMalloc(uma_memory_pool_handle_t hPool, size_t size) {
  return hPool->ops.malloc(hPool->pool_priv, size);
}

void *umaPoolAlignedMalloc(uma_memory_pool_handle_t hPool, size_t size,
                           size_t alignment) {
  return hPool->ops.aligned_malloc(hPool->pool_priv, size, alignment);
}

void *umaPoolCalloc(uma_memory_pool_handle_t hPool, size_t num, size_t size) {
  return hPool->ops.calloc(hPool->pool_priv, num, size);
}

void *umaPoolRealloc(uma_memory_pool_handle_t hPool, void *ptr, size_t size) {
  return hPool->ops.realloc(hPool->pool_priv, ptr, size);
}

size_t umaPoolMallocUsableSize(uma_memory_pool_handle_t hPool, void *ptr) {
  return hPool->ops.malloc_usable_size(hPool->pool_priv, ptr);
}

void umaPoolFree(uma_memory_pool_handle_t hPool, void *ptr) {
  hPool->ops.free(hPool->pool_priv, ptr);
}
