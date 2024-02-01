// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/raii.h"

using urCudaEventGetNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaEventGetNativeHandleTest);

TEST_P(urCudaEventGetNativeHandleTest, Success) {
    constexpr size_t buffer_size = 1024;
    uur::raii::Mem mem = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     buffer_size, nullptr, mem.ptr()));

    uur::raii::Event event = nullptr;
    uint8_t pattern = 6;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, mem, &pattern, sizeof(pattern),
                                          0, buffer_size, 0, nullptr,
                                          event.ptr()));

    ur_native_handle_t native_event = nullptr;
    ASSERT_SUCCESS(urEventGetNativeHandle(event, &native_event));
    CUevent cuda_event = reinterpret_cast<CUevent>(native_event);

    ASSERT_SUCCESS_CUDA(cuEventSynchronize(cuda_event));
}
