// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using urMemBufferPartitionTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferPartitionTest);

TEST_P(urMemBufferPartitionTest, Success) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              1024};
    uur::raii::Mem partition = nullptr;
    ASSERT_SUCCESS(urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                        UR_BUFFER_CREATE_TYPE_REGION, &region,
                                        partition.ptr()));
    ASSERT_NE(partition, nullptr);
}

TEST_P(urMemBufferPartitionTest, InvalidNullHandleBuffer) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              1024};
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemBufferPartition(nullptr, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_REGION, &region,
                                          partition.ptr()));
}

TEST_P(urMemBufferPartitionTest, InvalidEnumerationFlags) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              1024};
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemBufferPartition(buffer, UR_MEM_FLAG_FORCE_UINT32,
                                          UR_BUFFER_CREATE_TYPE_REGION, &region,
                                          partition.ptr()));
}

TEST_P(urMemBufferPartitionTest, InvalidEnumerationBufferCreateType) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              1024};
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_FORCE_UINT32,
                                          &region, partition.ptr()));
}

TEST_P(urMemBufferPartitionTest, InvalidNullPointerBufferCreateInfo) {
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_REGION, nullptr,
                                          partition.ptr()));
}

TEST_P(urMemBufferPartitionTest, InvalidNullPointerMem) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              1024};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_REGION, &region,
                                          nullptr));
}

TEST_P(urMemBufferPartitionTest, InvalidBufferSize) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0, 0};
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
                     urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_REGION, &region,
                                          partition.ptr()));
}

TEST_P(urMemBufferPartitionTest, InvalidValueCreateType) {
    // create a read only buffer
    uur::raii::Mem ro_buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, 4096,
                                     nullptr, ro_buffer.ptr()));

    // attempting to partition it into a RW buffer should fail
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              1024};
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                     urMemBufferPartition(ro_buffer, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_REGION, &region,
                                          partition.ptr()));
}

TEST_P(urMemBufferPartitionTest, InvalidValueBufferCreateInfoOutOfBounds) {
    ur_buffer_region_t region{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, 0,
                              8192};
    uur::raii::Mem partition = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
                     urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                          UR_BUFFER_CREATE_TYPE_REGION, &region,
                                          partition.ptr()));
}
