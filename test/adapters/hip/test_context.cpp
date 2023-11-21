// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "context.hpp"
#include "fixtures.h"
#include "queue.hpp"

using urHipContextTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urHipContextTest);

TEST_P(urHipContextTest, ActiveContexts) {
    ur_context_handle_t context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context));
    ASSERT_NE(context, nullptr);

    ur_queue_handle_t queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
    ASSERT_NE(queue, nullptr);

    // ensure that the queue has the correct context
    ASSERT_EQ(context, queue->getContext());

    // check that the current context is the active HIP context
    hipCtx_t hipContext = nullptr;
    ASSERT_SUCCESS_HIP(hipCtxGetCurrent(&hipContext));
    ASSERT_NE(hipContext, nullptr);
    ASSERT_EQ(hipContext, context->getDevice()->getNativeContext());

    ASSERT_SUCCESS(urQueueRelease(queue));
    ASSERT_SUCCESS(urContextRelease(context));
}

TEST_P(urHipContextTest, ActiveContextsThreads) {
    ur_context_handle_t context1 = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context1));
    ASSERT_NE(context1, nullptr);

    ur_context_handle_t context2 = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context2));
    ASSERT_NE(context2, nullptr);

    // setup synchronization
    std::mutex mtx;
    std::condition_variable cv;
    bool released = false;
    bool thread_done = false;

    auto test_thread = std::thread([&] {
        hipCtx_t current = nullptr;
        ur_queue_handle_t queue = nullptr;
        ASSERT_SUCCESS(urQueueCreate(context1, device, nullptr, &queue));
        ASSERT_NE(queue, nullptr);

        // ensure queue has the correct context
        ASSERT_EQ(queue->getContext(), context1);

        // check that the first context is now the active HIP context
        ASSERT_SUCCESS_HIP(hipCtxGetCurrent(&current));
        ASSERT_EQ(current, context1->getDevice()->getNativeContext());

        ASSERT_SUCCESS(urQueueRelease(queue));

        // mark the first set of processing as done and notify the main thread
        {
            std::unique_lock<std::mutex> lock(mtx);
            thread_done = true;
        }
        cv.notify_one();

        // wait for main thread to release the first context
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return released; });
        }

        // create a queue with the 2nd context
        queue = nullptr;
        ASSERT_SUCCESS(urQueueCreate(context2, device, nullptr, &queue));
        ASSERT_NE(queue, nullptr);

        // ensure the queue has the correct context
        ASSERT_EQ(queue->getContext(), context2);

        // check that the second context is now the active HIP context
        ASSERT_SUCCESS_HIP(hipCtxGetCurrent(&current));
        ASSERT_EQ(current, context2->getDevice()->getNativeContext());

        ASSERT_SUCCESS(urQueueRelease(queue));
    });

    // wait for the thread to be done with the first queue
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] { return thread_done; });
    ASSERT_SUCCESS(urContextRelease(context1));

    released = true;
    lock.unlock();
    cv.notify_one();

    // wait for thread to finish
    test_thread.join();

    ASSERT_SUCCESS(urContextRelease(context2));
}
#pragma GCC diagnostic pop
