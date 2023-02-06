// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_UMA_TEST_PROVIDER_H
#define UR_UMA_TEST_PROVIDER_H

#include <uma/memory_provider.h>

#if defined(__cplusplus)
extern "C" {
#endif

uma_memory_provider_handle_t nullProviderCreate(void);
uma_memory_provider_handle_t
traceProviderCreate(uma_memory_provider_handle_t hUpstreamProvider,
                    void (*trace)(const char *));
uma_memory_provider_handle_t mallocProviderCreate(void);

#if defined(__cplusplus)
}
#endif

#endif // UR_UMA_TEST_PROVIDER_H
