// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urMemBufferCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferCreateTest);

TEST_P(urMemBufferCreateTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 4096,
                                     nullptr, &buffer));
    ASSERT_NE(nullptr, buffer);
    ASSERT_SUCCESS(urMemRelease(buffer));
}

TEST_P(urMemBufferCreateTest, InvalidNullHandleContext) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemBufferCreate(nullptr, UR_MEM_FLAG_READ_WRITE, 4096,
                                       nullptr, &buffer));
}

TEST_P(urMemBufferCreateTest, InvalidEnumerationFlags) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemBufferCreate(context, UR_MEM_FLAG_FORCE_UINT32, 4096,
                                       nullptr, &buffer));
}

using urMemBufferCreateTestWithFlagsParam =
    uur::urContextTestWithParam<ur_mem_flag_t>;

using urMemBufferCreateWithHostPtrFlagsTest =
    urMemBufferCreateTestWithFlagsParam;
UUR_TEST_SUITE_P(urMemBufferCreateWithHostPtrFlagsTest,
                 ::testing::Values(UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER,
                                   UR_MEM_FLAG_ALLOC_HOST_POINTER,
                                   UR_MEM_FLAG_USE_HOST_POINTER),
                 uur::deviceTestWithParamPrinter<ur_mem_flag_t>);

TEST_P(urMemBufferCreateWithHostPtrFlagsTest, InvalidHostPtr) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_HOST_PTR,
        urMemBufferCreate(context, getParam(), 4096, nullptr, &buffer));
}

TEST_P(urMemBufferCreateTest, InvalidNullPointerBuffer) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 4096,
                                       nullptr, nullptr));
}

TEST_P(urMemBufferCreateTest, InvalidBufferSizeZero) {
    ur_mem_handle_t buffer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
                     urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 0,
                                       nullptr, &buffer));
}

TEST_P(urMemBufferCreateTest, InvalidBufferSizeMax) {
    ur_mem_handle_t buffer = nullptr;
    uint64_t max_size = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                                   sizeof(uint64_t), &max_size, nullptr));
    ASSERT_NE(max_size, 0);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
                     urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                       max_size + 1, nullptr, &buffer));
}
