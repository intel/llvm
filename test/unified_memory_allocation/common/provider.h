// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_UMA_TEST_PROVIDER_H
#define UR_UMA_TEST_PROVIDER_H

#include <uma/memory_provider.h>

#if defined(__cplusplus)
extern "C" {
#endif

uma_memory_provider_handle_t nullProviderCreate();
uma_memory_provider_handle_t
traceProviderCreate(uma_memory_provider_handle_t hUpstreamProvider,
                    void (*trace)(const char *));

#if defined(__cplusplus)
}
#endif

#endif // UR_UMA_TEST_PROVIDER_H
