// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueMemBufferMapTest = uur::urMemBufferQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferMapTest);

TEST_P(urEnqueueMemBufferMapTest, SuccessRead) {
    const std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    uint32_t *map = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true, UR_MAP_FLAG_READ,
                                         0, size, 0, nullptr, nullptr,
                                         (void **)&map));
    for (unsigned i = 0; i < count; ++i) {
        ASSERT_EQ(map[i], 42) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferMapTest, SuccessWrite) {
    const std::vector<uint32_t> input(count, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    uint32_t *map = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true, UR_MAP_FLAG_WRITE,
                                         0, size, 0, nullptr, nullptr,
                                         (void **)&map));
    for (unsigned i = 0; i < count; ++i) {
        map[i] = 42;
    }
    ASSERT_SUCCESS(urEnqueueMemUnmap(queue, buffer, map, 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (unsigned i = 0; i < count; ++i) {
        ASSERT_EQ(output[i], 42) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferMapTest, SuccessOffset) {
    const std::vector<uint32_t> input(count, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    uint32_t *map = nullptr;
    const size_t offset_size = size / 2;
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true, UR_MAP_FLAG_WRITE,
                                         offset_size, size - offset_size, 0,
                                         nullptr, nullptr, (void **)&map));

    const size_t offset_count = count / 2;
    for (size_t i = 0; i < offset_count; ++i) {
        map[i] = 42;
    }

    ASSERT_SUCCESS(urEnqueueMemUnmap(queue, buffer, map, 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));

    for (size_t i = 0; i < offset_count; ++i) {
        ASSERT_EQ(output[i], 0) << "Result mismatch at index: " << i;
    }
    for (size_t i = offset_count; i < count; ++i) {
        ASSERT_EQ(output[i], 42) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferMapTest, SuccessPartialMap) {
    const std::vector<uint32_t> input(count, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
    uint32_t *map = nullptr;
    const size_t map_size = size / 2;
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true, UR_MAP_FLAG_WRITE,
                                         0, map_size, 0, nullptr, nullptr,
                                         (void **)&map));

    const size_t offset_count = count / 2;
    for (unsigned i = 0; i < offset_count; ++i) {
        map[i] = 42;
    }

    ASSERT_SUCCESS(urEnqueueMemUnmap(queue, buffer, map, 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));

    for (size_t i = 0; i < offset_count; ++i) {
        ASSERT_EQ(output[i], 42) << "Result mismatch at index: " << i;
    }
    for (size_t i = offset_count; i < count; ++i) {
        ASSERT_EQ(output[i], 0) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferMapTest, SuccesPinnedRead) {
    const size_t memSize = sizeof(int);
    const int value = 20;

    ur_mem_handle_t memObj;
    ASSERT_SUCCESS(urMemBufferCreate(
        context, UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_ALLOC_HOST_POINTER,
        memSize, nullptr, &memObj));

    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, memObj, true, 0, sizeof(int),
                                           &value, 0, nullptr, nullptr));

    int *host_ptr = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, memObj, true, UR_MAP_FLAG_READ,
                                         0, sizeof(int), 0, nullptr, nullptr,
                                         (void **)&host_ptr));

    ASSERT_EQ(*host_ptr, value);
    ASSERT_SUCCESS(
        urEnqueueMemUnmap(queue, memObj, host_ptr, 0, nullptr, nullptr));

    ASSERT_SUCCESS(urMemRelease(memObj));
}

TEST_P(urEnqueueMemBufferMapTest, SuccesPinnedWrite) {
    const size_t memSize = sizeof(int);
    const int value = 30;

    ur_mem_handle_t memObj;
    ASSERT_SUCCESS(urMemBufferCreate(
        context, UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_ALLOC_HOST_POINTER,
        memSize, nullptr, &memObj));

    int *host_ptr = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, memObj, true, UR_MAP_FLAG_WRITE,
                                         0, sizeof(int), 0, nullptr, nullptr,
                                         (void **)&host_ptr));

    *host_ptr = value;

    ASSERT_SUCCESS(
        urEnqueueMemUnmap(queue, memObj, host_ptr, 0, nullptr, nullptr));

    int read_value = 0;
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, memObj, true, 0, sizeof(int),
                                          &read_value, 0, nullptr, nullptr));

    ASSERT_EQ(read_value, value);
    ASSERT_SUCCESS(urMemRelease(memObj));
}

TEST_P(urEnqueueMemBufferMapTest, SuccessMultiMaps) {
    const std::vector<uint32_t> input(count, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    // Create two maps with non-overlapping ranges and write separate values
    // into each of them to check we can maintain multiple maps on the same
    // buffer.
    uint32_t *map_a = nullptr;
    uint32_t *map_b = nullptr;
    const auto map_size = size / 2;
    const auto map_offset = size / 2;
    const auto map_count = count / 2;

    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true, UR_MAP_FLAG_WRITE,
                                         0, map_size, 0, nullptr, nullptr,
                                         (void **)&map_a));
    ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true, UR_MAP_FLAG_WRITE,
                                         map_offset, map_size, 0, nullptr,
                                         nullptr, (void **)&map_b));
    for (size_t i = 0; i < map_count; ++i) {
        map_a[i] = 42;
    }
    for (size_t i = map_count; i < count; ++i) {
        map_a[i] = 24;
    }
    ASSERT_SUCCESS(
        urEnqueueMemUnmap(queue, buffer, map_a, 0, nullptr, nullptr));
    ASSERT_SUCCESS(
        urEnqueueMemUnmap(queue, buffer, map_b, 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (size_t i = 0; i < map_count; ++i) {
        ASSERT_EQ(output[i], 42) << "Result mismatch at index: " << i;
    }
    for (size_t i = map_count; i < count; ++i) {
        ASSERT_EQ(output[i], 24) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferMapTest, InvalidNullHandleQueue) {
    void *map = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferMap(nullptr, buffer, true,
                                           UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE,
                                           0, size, 0, nullptr, nullptr, &map));
}

TEST_P(urEnqueueMemBufferMapTest, InvalidNullHandleBuffer) {
    void *map = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferMap(queue, nullptr, true,
                                           UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE,
                                           0, size, 0, nullptr, nullptr, &map));
}

TEST_P(urEnqueueMemBufferMapTest, InvalidEnumerationMapFlags) {
    void *map = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urEnqueueMemBufferMap(queue, buffer, true,
                                           UR_MAP_FLAG_FORCE_UINT32, 0, size, 0,
                                           nullptr, nullptr, &map));
}

TEST_P(urEnqueueMemBufferMapTest, InvalidNullPointerRetMap) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemBufferMap(queue, buffer, true,
                                           UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE,
                                           0, size, 0, nullptr, nullptr,
                                           nullptr));
}

TEST_P(urEnqueueMemBufferMapTest, InvalidNullPtrEventWaitList) {
    void *map;
    ASSERT_EQ_RESULT(urEnqueueMemBufferMap(queue, buffer, true,
                                           UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE,
                                           0, size, 1, nullptr, nullptr, &map),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemBufferMap(queue, buffer, true,
                                           UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE,
                                           0, size, 0, &validEvent, nullptr,
                                           &map),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemBufferMap(queue, buffer, true,
                                           UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE,
                                           0, size, 1, &inv_evt, nullptr, &map),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferMapTest, InvalidSize) {
    void *map = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemBufferMap(queue, buffer, true, 0, 1, size, 0,
                                           nullptr, nullptr, &map));
}

using urEnqueueMemBufferMapMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;

TEST_F(urEnqueueMemBufferMapMultiDeviceTest, WriteMapDifferentQueues) {
    // First queue does a blocking write of 42 into the buffer.
    std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    // Then the remaining queues map the buffer into some host memory. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        uint32_t *map = nullptr;
        ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer, true,
                                             UR_MAP_FLAG_READ, 0, size, 0,
                                             nullptr, nullptr, (void **)&map));
        for (unsigned j = 0; j < count; ++j) {
            ASSERT_EQ(input[j], map[j])
                << "Result on queue " << i << " at index " << j
                << " did not match!";
        }
        ASSERT_SUCCESS(
            urEnqueueMemUnmap(queue, buffer, map, 0, nullptr, nullptr));
    }
}
