/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <uma/memory_provider.h>

#include <assert.h>
#include <stdlib.h>

struct uma_memory_provider_t {
  struct uma_memory_provider_ops_t ops;
  void *provider_priv;
};

enum uma_result_t
umaMemoryProviderCreate(struct uma_memory_provider_ops_t *ops, void *params,
                        uma_memory_provider_handle_t *hProvider) {
  uma_memory_provider_handle_t provider =
      malloc(sizeof(struct uma_memory_provider_t));
  if (!provider) {
    return UMA_RESULT_RUNTIME_ERROR;
  }

  assert(ops->version == UMA_VERSION_CURRENT);

  provider->ops = *ops;

  void *provider_priv;
  enum uma_result_t ret = ops->initialize(params, &provider_priv);
  if (ret != UMA_RESULT_SUCCESS) {
    return ret;
  }

  provider->provider_priv = provider_priv;

  *hProvider = provider;

  return UMA_RESULT_SUCCESS;
}

void umaMemoryProviderDestroy(uma_memory_provider_handle_t hProvider) {
  hProvider->ops.finalize(hProvider->provider_priv);
  free(hProvider);
}

enum uma_result_t umaMemoryProviderAlloc(uma_memory_provider_handle_t hProvider,
                                         size_t size, size_t alignment,
                                         void **ptr) {
  return hProvider->ops.alloc(hProvider->provider_priv, size, alignment, ptr);
}

enum uma_result_t umaMemoryProviderFree(uma_memory_provider_handle_t hProvider,
                                        void *ptr, size_t size) {
  return hProvider->ops.free(hProvider->provider_priv, ptr, size);
}
