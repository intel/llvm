// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_UMF_TEST_PROVIDER_H
#define UR_UMF_TEST_PROVIDER_H

#include <umf/memory_provider.h>

#if defined(__cplusplus)
extern "C" {
#endif

umf_memory_provider_handle_t nullProviderCreate(void);
umf_memory_provider_handle_t
traceProviderCreate(umf_memory_provider_handle_t hUpstreamProvider,
                    void (*trace)(const char *));

#if defined(__cplusplus)
}
#endif

#endif // UR_UMF_TEST_PROVIDER_H
