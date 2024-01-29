// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urMemBufferCreateTestWithFlagsParam =
    uur::urContextTestWithParam<ur_mem_flag_t>;

using urMemBufferCreateWithFlagsTest = urMemBufferCreateTestWithFlagsParam;
UUR_TEST_SUITE_P(urMemBufferCreateWithFlagsTest,
                 ::testing::Values(UR_MEM_FLAG_READ_WRITE,
                                   UR_MEM_FLAG_WRITE_ONLY,
                                   UR_MEM_FLAG_READ_ONLY,
                                   UR_MEM_FLAG_ALLOC_HOST_POINTER),
                 uur::deviceTestWithParamPrinter<ur_mem_flag_t>);

TEST_P(urMemBufferCreateWithFlagsTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_SUCCESS(
        urMemBufferCreate(context, getParam(), 4096, nullptr, &buffer));
    ASSERT_NE(nullptr, buffer);
    ASSERT_SUCCESS(urMemRelease(buffer));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidNullHandleContext) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urMemBufferCreate(nullptr, getParam(), 4096, nullptr, &buffer));
}

using urMemBufferCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferCreateTest);

TEST_P(urMemBufferCreateTest, InvalidEnumerationFlags) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemBufferCreate(context, UR_MEM_FLAG_FORCE_UINT32, 4096,
                                       nullptr, &buffer));
}

using urMemBufferCreateWithHostPtrFlagsTest =
    urMemBufferCreateTestWithFlagsParam;
UUR_TEST_SUITE_P(urMemBufferCreateWithHostPtrFlagsTest,
                 ::testing::Values(UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER,
                                   UR_MEM_FLAG_USE_HOST_POINTER),
                 uur::deviceTestWithParamPrinter<ur_mem_flag_t>);

TEST_P(urMemBufferCreateWithHostPtrFlagsTest, InvalidHostPtr) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_HOST_PTR,
        urMemBufferCreate(context, getParam(), 4096, nullptr, &buffer));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidNullPointerBuffer) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urMemBufferCreate(context, getParam(), 4096, nullptr, nullptr));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidBufferSizeZero) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
        urMemBufferCreate(context, getParam(), 0, nullptr, &buffer));
}
