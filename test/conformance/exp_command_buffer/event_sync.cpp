// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

// Tests non-kernel commands using ur events for synchronization work as expected
using CommandEventSyncTest = uur::command_buffer::urCommandEventSyncTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(CommandEventSyncTest);

TEST_P(CommandEventSyncTest, USMMemcpyExp) {
    // Get wait event from queue fill on ptr 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                    &patternX, allocation_size, 0, nullptr,
                                    &external_events[0]));

    // Command to fill ptr 1
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[1], &patternY, sizeof(patternY),
        allocation_size, 0, nullptr, 0, nullptr, &sync_points[0], nullptr,
        nullptr));

    // Test command overwriting ptr 1 with ptr 0 command based on queue event
    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        cmd_buf_handle, device_ptrs[1], device_ptrs[0], allocation_size, 1,
        &sync_points[0], 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read ptr 1 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                      device_ptrs[1], allocation_size, 1,
                                      &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternX);
    }
}

TEST_P(CommandEventSyncTest, USMFillExp) {
    // Get wait event from queue fill on ptr 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                    &patternX, allocation_size, 0, nullptr,
                                    &external_events[0]));

    // Test fill command overwriting ptr 0 waiting on queue event
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[0], &patternY, sizeof(patternY),
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read ptr 0 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                      device_ptrs[0], allocation_size, 1,
                                      &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternY);
    }
}

// Test fill using a large pattern size since implementations may need to handle
// this differently.
TEST_P(CommandEventSyncTest, USMFillLargePatternExp) {
    // Device ptrs are allocated in the test fixture with 32-bit values * num
    // elements, since we are doubling the pattern size we want to treat those
    // device pointers as if they were created with half the number of elements.
    constexpr size_t modifiedElementSize = elements / 2;
    // Get wait event from queue fill on ptr 0
    uint64_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                    &patternX, allocation_size, 0, nullptr,
                                    &external_events[0]));

    // Test fill command overwriting ptr 0 waiting on queue event
    uint64_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[0], &patternY, sizeof(patternY),
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read ptr 0 based on event returned from command-buffer command
    std::array<uint64_t, modifiedElementSize> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                      device_ptrs[0], allocation_size, 1,
                                      &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < modifiedElementSize; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternY);
    }
}

TEST_P(CommandEventSyncTest, MemBufferCopyExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Command to fill buffer 1
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
        cmd_buf_handle, buffers[1], &patternY, sizeof(patternY), 0,
        allocation_size, 0, nullptr, 0, nullptr, &sync_points[0], nullptr,
        nullptr));

    // Test command overwriting buffer 1 with buffer 0 command based on queue event
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyExp(
        cmd_buf_handle, buffers[0], buffers[1], 0, 0, allocation_size, 1,
        &sync_points[0], 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read buffer 1 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[1], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternX);
    }
}

TEST_P(CommandEventSyncTest, MemBufferCopyRectExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Command to fill buffer 1
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
        cmd_buf_handle, buffers[1], &patternY, sizeof(patternY), 0,
        allocation_size, 0, nullptr, 0, nullptr, &sync_points[0], nullptr,
        nullptr));

    // Test command overwriting buffer 1 with buffer 0 command based on queue event
    ur_rect_offset_t src_origin{0, 0, 0};
    ur_rect_offset_t dst_origin{0, 0, 0};
    constexpr size_t rect_buffer_row_size = 16;
    ur_rect_region_t region{rect_buffer_row_size, rect_buffer_row_size, 1};
    size_t src_row_pitch = rect_buffer_row_size;
    size_t src_slice_pitch = allocation_size;
    size_t dst_row_pitch = rect_buffer_row_size;
    size_t dst_slice_pitch = allocation_size;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyRectExp(
        cmd_buf_handle, buffers[0], buffers[1], src_origin, dst_origin, region,
        src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, 1,
        &sync_points[0], 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read buffer 1 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[1], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternX);
    }
}

TEST_P(CommandEventSyncTest, MemBufferReadExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Test command reading buffer 0 based on queue event
    std::array<uint32_t, elements> host_command_ptr{};
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadExp(
        cmd_buf_handle, buffers[0], 0, allocation_size, host_command_ptr.data(),
        0, nullptr, 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Overwrite buffer 0 based on event returned from command-buffer command,
    // then read back to verify ordering
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(
        queue, buffers[0], &patternY, sizeof(patternY), 0, allocation_size, 1,
        &external_events[1], &external_events[2]));
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[0], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[2], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_command_ptr[i], patternX);
        ASSERT_EQ(host_enqueue_ptr[i], patternY);
    }
}

TEST_P(CommandEventSyncTest, MemBufferReadRectExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Test command reading buffer 0 based on queue event
    std::array<uint32_t, elements> host_command_ptr{};
    ur_rect_offset_t buffer_offset = {0, 0, 0};
    ur_rect_offset_t host_offset = {0, 0, 0};
    constexpr size_t rect_buffer_row_size = 16;
    ur_rect_region_t region = {rect_buffer_row_size, rect_buffer_row_size, 1};
    size_t buffer_row_pitch = rect_buffer_row_size;
    size_t buffer_slice_pitch = allocation_size;
    size_t host_row_pitch = rect_buffer_row_size;
    size_t host_slice_pitch = allocation_size;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadRectExp(
        cmd_buf_handle, buffers[0], buffer_offset, host_offset, region,
        buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
        host_command_ptr.data(), 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Overwrite buffer 0 based on event returned from command-buffer command,
    // then read back to verify ordering
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(
        queue, buffers[0], &patternY, sizeof(patternY), 0, allocation_size, 1,
        &external_events[1], &external_events[2]));
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[0], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[2], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_command_ptr[i], patternX);
        ASSERT_EQ(host_enqueue_ptr[i], patternY);
    }
}

TEST_P(CommandEventSyncTest, MemBufferWriteExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Test command overwriting buffer 0 based on queue event
    std::array<uint32_t, elements> host_command_ptr{};
    uint32_t patternY = 0xA;
    std::fill(host_command_ptr.begin(), host_command_ptr.end(), patternY);
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteExp(
        cmd_buf_handle, buffers[0], 0, allocation_size, host_command_ptr.data(),
        0, nullptr, 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Read back buffer 0 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[0], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternY) << i;
    }
}

TEST_P(CommandEventSyncTest, MemBufferWriteRectExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Test command overwriting buffer 0 based on queue event
    std::array<uint32_t, elements> host_command_ptr{};
    uint32_t patternY = 0xA;
    std::fill(host_command_ptr.begin(), host_command_ptr.end(), patternY);

    ur_rect_offset_t buffer_offset = {0, 0, 0};
    ur_rect_offset_t host_offset = {0, 0, 0};
    constexpr size_t rect_buffer_row_size = 16;
    ur_rect_region_t region = {rect_buffer_row_size, rect_buffer_row_size, 1};
    size_t buffer_row_pitch = rect_buffer_row_size;
    size_t buffer_slice_pitch = allocation_size;
    size_t host_row_pitch = rect_buffer_row_size;
    size_t host_slice_pitch = allocation_size;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteRectExp(
        cmd_buf_handle, buffers[0], buffer_offset, host_offset, region,
        buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
        host_command_ptr.data(), 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Read back buffer 0 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[0], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternY) << i;
    }
}

TEST_P(CommandEventSyncTest, MemBufferFillExp) {
    // Get wait event from queue fill on buffer 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Test fill command overwriting buffer 0 based on queue event
    uint32_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
        cmd_buf_handle, buffers[0], &patternY, sizeof(patternY), 0,
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read buffer 0 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[0], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternY);
    }
}

// Test fill using a large pattern size since implementations may need to handle
// this differently.
TEST_P(CommandEventSyncTest, MemBufferFillLargePatternExp) {
    // Device buffers are allocated in the test fixture with 32-bit values * num
    // elements, since we are doubling the pattern size we want to treat those
    // device pointers as if they were created with half the number of elements.
    constexpr size_t modifiedElementSize = elements / 2;
    // Get wait event from queue fill on buffer 0
    uint64_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffers[0], &patternX,
                                          sizeof(patternX), 0, allocation_size,
                                          0, nullptr, &external_events[0]));

    // Test fill command overwriting buffer 0 based on queue event
    uint64_t patternY = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
        cmd_buf_handle, buffers[0], &patternY, sizeof(patternY), 0,
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read buffer 0 based on event returned from command-buffer command
    std::array<uint64_t, modifiedElementSize> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffers[0], false, 0, allocation_size, host_enqueue_ptr.data(),
        1, &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < modifiedElementSize; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternY);
    }
}

TEST_P(CommandEventSyncTest, USMPrefetchExp) {
    // Get wait event from queue fill on ptr 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                    &patternX, allocation_size, 0, nullptr,
                                    &external_events[0]));

    // Test prefetch command waiting on queue event
    ASSERT_SUCCESS(urCommandBufferAppendUSMPrefetchExp(
        cmd_buf_handle, device_ptrs[1], allocation_size, 0 /* migration flags*/,
        0, nullptr, 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read ptr 0 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                      device_ptrs[0], allocation_size, 1,
                                      &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternX);
    }
}

TEST_P(CommandEventSyncTest, USMAdviseExp) {
    // Get wait event from queue fill on ptr 0
    uint32_t patternX = 42;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                    &patternX, allocation_size, 0, nullptr,
                                    &external_events[0]));

    // Test advise command waiting on queue event
    ASSERT_SUCCESS(urCommandBufferAppendUSMAdviseExp(
        cmd_buf_handle, device_ptrs[0], allocation_size, 0 /* advice flags*/, 0,
        nullptr, 1, &external_events[0], nullptr, &external_events[1],
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read ptr 0 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptr{};
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                      device_ptrs[0], allocation_size, 1,
                                      &external_events[1], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptr[i], patternX);
    }
}

TEST_P(CommandEventSyncTest, MultipleEventCommands) {
    // Command to fill ptr 0
    uint32_t patternA = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[0], &patternA, sizeof(patternA),
        allocation_size, 0, nullptr, 0, nullptr, nullptr, &external_events[0],
        nullptr));

    // Command to fill ptr 1
    uint32_t patternB = 0xB;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[1], &patternB, sizeof(patternB),
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));

    // Command to fill ptr 1
    uint32_t patternC = 0xC;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[2], &patternC, sizeof(patternC),
        allocation_size, 0, nullptr, 1, &external_events[1], nullptr,
        &external_events[2], nullptr));

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

    // Queue read ptr 1 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptrA, host_enqueue_ptrB,
        host_enqueue_ptrC;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptrA.data(),
                                      device_ptrs[0], allocation_size, 1,
                                      &external_events[0], nullptr));

    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptrB.data(),
                                      device_ptrs[1], allocation_size, 1,
                                      &external_events[1], nullptr));

    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptrC.data(),
                                      device_ptrs[2], allocation_size, 1,
                                      &external_events[2], nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptrA[i], patternA);
        ASSERT_EQ(host_enqueue_ptrB[i], patternB);
        ASSERT_EQ(host_enqueue_ptrC[i], patternC);
    }
}

TEST_P(CommandEventSyncTest, MultipleEventCommandsBetweenCommandBuffers) {
    // Command to fill ptr 0
    uint32_t patternA = 0xA;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[0], &patternA, sizeof(patternA),
        allocation_size, 0, nullptr, 0, nullptr, nullptr, &external_events[0],
        nullptr));

    // Command to fill ptr 1
    uint32_t patternB = 0xB;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[1], &patternB, sizeof(patternB),
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr,
        &external_events[1], nullptr));

    // Command to fill ptr 1
    uint32_t patternC = 0xC;
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptrs[2], &patternC, sizeof(patternC),
        allocation_size, 0, nullptr, 1, &external_events[1], nullptr,
        &external_events[2], nullptr));

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

    // Queue read ptr 1 based on event returned from command-buffer command
    std::array<uint32_t, elements> host_enqueue_ptrA, host_enqueue_ptrB,
        host_enqueue_ptrC;
    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        second_cmd_buf_handle, host_enqueue_ptrA.data(), device_ptrs[0],
        allocation_size, 0, nullptr, 1, &external_events[0], nullptr, nullptr,
        nullptr));

    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        second_cmd_buf_handle, host_enqueue_ptrB.data(), device_ptrs[1],
        allocation_size, 0, nullptr, 1, &external_events[1], nullptr, nullptr,
        nullptr));

    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        second_cmd_buf_handle, host_enqueue_ptrC.data(), device_ptrs[2],
        allocation_size, 0, nullptr, 1, &external_events[2], nullptr, nullptr,
        nullptr));

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(second_cmd_buf_handle));
    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(second_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));

    // Verify
    ASSERT_SUCCESS(urQueueFinish(queue));
    for (size_t i = 0; i < elements; i++) {
        ASSERT_EQ(host_enqueue_ptrA[i], patternA);
        ASSERT_EQ(host_enqueue_ptrB[i], patternB);
        ASSERT_EQ(host_enqueue_ptrC[i], patternC);
    }
}
