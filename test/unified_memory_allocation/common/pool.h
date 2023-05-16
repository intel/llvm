// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_UMA_TEST_POOL_H
#define UR_UMA_TEST_POOL_H

#include <uma/memory_pool.h>

#if defined(__cplusplus)
extern "C" {
#endif

uma_memory_pool_handle_t nullPoolCreate(void);
uma_memory_pool_handle_t
tracePoolCreate(uma_memory_pool_handle_t hUpstreamPool,
                uma_memory_provider_handle_t providerDesc,
                void (*trace)(const char *));

#if defined(__cplusplus)
}
#endif

#endif // UR_UMA_TEST_POOL_H
