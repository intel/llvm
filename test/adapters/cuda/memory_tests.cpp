// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/raii.h"

using cudaMemoryTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(cudaMemoryTest);

TEST_P(cudaMemoryTest, urMemBufferNoActiveContext) {
    constexpr size_t memSize = 1024u;

    CUcontext current = nullptr;
    do {
        CUcontext oldContext = nullptr;
        ASSERT_SUCCESS_CUDA(cuCtxPopCurrent(&oldContext));
        ASSERT_SUCCESS_CUDA(cuCtxGetCurrent(&current));
    } while (current != nullptr);

    uur::raii::Mem mem;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, memSize,
                                     nullptr, mem.ptr()));
    ASSERT_NE(mem, nullptr);
}
