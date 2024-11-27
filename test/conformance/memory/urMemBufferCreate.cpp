// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

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
    uur::raii::Mem buffer = nullptr;
    ASSERT_SUCCESS(
        urMemBufferCreate(context, getParam(), 4096, nullptr, buffer.ptr()));
    ASSERT_NE(nullptr, buffer);
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidNullHandleContext) {
    uur::raii::Mem buffer = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urMemBufferCreate(nullptr, getParam(), 4096, nullptr, buffer.ptr()));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidNullPointerBuffer) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urMemBufferCreate(context, getParam(), 4096, nullptr, nullptr));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidBufferSizeZero) {
    uur::raii::Mem buffer = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
        urMemBufferCreate(context, getParam(), 0, nullptr, buffer.ptr()));
}

using urMemBufferCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferCreateTest);

TEST_P(urMemBufferCreateTest, InvalidEnumerationFlags) {
    uur::raii::Mem buffer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemBufferCreate(context, UR_MEM_FLAG_FORCE_UINT32, 4096,
                                       nullptr, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, InvalidHostPtrNullProperties) {
    uur::raii::Mem buffer = nullptr;
    ur_mem_flags_t flags =
        UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_HOST_PTR,
        urMemBufferCreate(context, flags, 4096, nullptr, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, InvalidHostPtrNullHost) {
    uur::raii::Mem buffer = nullptr;
    ur_mem_flags_t flags =
        UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
    ur_buffer_properties_t properties;
    properties.pHost = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_HOST_PTR,
        urMemBufferCreate(context, flags, 4096, &properties, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, InvalidHostPtrValidHost) {
    uur::raii::Mem buffer = nullptr;
    ur_mem_flags_t flags = 0;
    ur_buffer_properties_t properties;
    int data = 42;
    properties.pHost = &data;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_HOST_PTR,
        urMemBufferCreate(context, flags, 4096, &properties, buffer.ptr()));
}

using urMemBufferCreateWithHostPtrFlagsTest =
    urMemBufferCreateTestWithFlagsParam;
UUR_TEST_SUITE_P(urMemBufferCreateWithHostPtrFlagsTest,
                 ::testing::Values(UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER,
                                   UR_MEM_FLAG_USE_HOST_POINTER),
                 uur::deviceTestWithParamPrinter<ur_mem_flag_t>);

TEST_P(urMemBufferCreateWithHostPtrFlagsTest, SUCCESS) {
    uur::raii::Mem host_ptr_buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_ALLOC_HOST_POINTER,
                                     4096, nullptr, host_ptr_buffer.ptr()));

    ur_buffer_properties_t properties{UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                      nullptr, host_ptr_buffer.ptr()};
    uur::raii::Mem buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, getParam(), 4096, &properties,
                                     buffer.ptr()));
}
