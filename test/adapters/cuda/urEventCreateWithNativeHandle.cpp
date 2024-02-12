// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/raii.h"

using urCudaEventCreateWithNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaEventCreateWithNativeHandleTest);

struct RAIICUevent {
    CUevent handle = nullptr;

    ~RAIICUevent() {
        if (handle) {
            cuEventDestroy(handle);
        }
    }

    CUevent *ptr() { return &handle; }
    CUevent get() { return handle; }
};

TEST_P(urCudaEventCreateWithNativeHandleTest, Success) {
    RAIICUevent cuda_event;
    ASSERT_SUCCESS_CUDA(cuEventCreate(cuda_event.ptr(), CU_EVENT_DEFAULT));

    ur_native_handle_t native_event =
        reinterpret_cast<ur_native_handle_t>(cuda_event.get());

    uur::raii::Event event = nullptr;
    EXPECT_SUCCESS(urEventCreateWithNativeHandle(native_event, context, nullptr,
                                                 event.ptr()));
}
