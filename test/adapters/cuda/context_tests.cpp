// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "context.hpp"
#include "fixtures.h"
#include "queue.hpp"
#include "uur/raii.h"
#include <thread>

using cudaUrContextCreateTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(cudaUrContextCreateTest);

constexpr unsigned int known_cuda_api_version = 3020;

TEST_P(cudaUrContextCreateTest, CreateWithChildThread) {
    uur::raii::Context context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
    ASSERT_NE(context, nullptr);

    // Retrieve the CUDA context to check information is correct
    auto checkValue = [=] {
        CUcontext cudaContext = context.handle->get();
        unsigned int version = 0;
        EXPECT_SUCCESS_CUDA(cuCtxGetApiVersion(cudaContext, &version));
        EXPECT_EQ(version, known_cuda_api_version);

        // The current context is different from the current thread
        CUcontext current;
        ASSERT_SUCCESS_CUDA(cuCtxGetCurrent(&current));
        EXPECT_NE(cudaContext, current);

        // Set the context
        EXPECT_SUCCESS_CUDA(cuCtxPushCurrent(cudaContext));
        EXPECT_SUCCESS_CUDA(cuCtxGetCurrent(&current));
        EXPECT_EQ(cudaContext, current);
    };

    auto callContextFromOtherThread = std::thread(checkValue);
    callContextFromOtherThread.join();
}

TEST_P(cudaUrContextCreateTest, ActiveContext) {
    uur::raii::Context context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
    ASSERT_NE(context, nullptr);

    uur::raii::Queue queue = nullptr;
    ur_queue_properties_t queue_props{UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                      nullptr, 0};
    ASSERT_SUCCESS(urQueueCreate(context, device, &queue_props, queue.ptr()));
    ASSERT_NE(queue, nullptr);

    // check that the queue has the correct context
    ASSERT_EQ(context, queue->getContext());

    // create a buffer
    uur::raii::Mem buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 1024,
                                     nullptr, buffer.ptr()));
    ASSERT_NE(buffer, nullptr);

    // check that the context is now the active CUDA context
    CUcontext cudaCtx = nullptr;
    ASSERT_SUCCESS_CUDA(cuCtxGetCurrent(&cudaCtx));
    ASSERT_NE(cudaCtx, nullptr);

    ur_native_handle_t native_context = nullptr;
    ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_context));
    ASSERT_NE(native_context, nullptr);
    ASSERT_EQ(cudaCtx, reinterpret_cast<CUcontext>(native_context));
}

TEST_P(cudaUrContextCreateTest, ContextLifetimeExisting) {
    // start by setting up a CUDA context on the thread
    CUcontext original;
    ASSERT_SUCCESS_CUDA(cuCtxCreate(&original, CU_CTX_MAP_HOST, device->get()));

    // ensure the CUDA context is active
    CUcontext current = nullptr;
    ASSERT_SUCCESS_CUDA(cuCtxGetCurrent(&current));
    ASSERT_EQ(original, current);

    // create a UR context
    uur::raii::Context context;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
    ASSERT_NE(context, nullptr);

    // create a queue with the context
    uur::raii::Queue queue;
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, queue.ptr()));
    ASSERT_NE(queue, nullptr);

    // ensure the queue has the correct context
    ASSERT_EQ(context, queue->getContext());

    // create a buffer in the context to set the context as active
    ur_mem_handle_t buffer;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 1024,
                                     nullptr, &buffer));

    // check that context is now the active cuda context
    ASSERT_SUCCESS_CUDA(cuCtxGetCurrent(&current));
    ASSERT_EQ(current, context->get());
}

TEST_P(cudaUrContextCreateTest, ThreadedContext) {
    // create two new UR contexts
    uur::raii::Context context1;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context1.ptr()));
    ASSERT_NE(context1, nullptr);

    uur::raii::Context context2;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context2.ptr()));
    ASSERT_NE(context2, nullptr);

    // setup synchronization variables between the main thread and
    // the testing thread
    std::mutex m;
    std::condition_variable cv;
    bool released = false;
    bool thread_done = false;

    // create a testing thread that will create a queue with the first context,
    // release the queue, then wait for the main thread to release
    // the first context, and then create and release another queue with
    // the second context.
    auto test_thread = std::thread([&] {
        CUcontext current = nullptr;

        {
            // create a queue with the first context
            uur::raii::Queue queue;
            ASSERT_SUCCESS(
                urQueueCreate(context1, device, nullptr, queue.ptr()));
            ASSERT_NE(queue, nullptr);

            // ensure that the queue has the correct context
            ASSERT_EQ(context1, queue->getContext());

            // create a buffer to set context1 as the active context
            uur::raii::Mem buffer;
            ASSERT_SUCCESS(urMemBufferCreate(context1, UR_MEM_FLAG_READ_WRITE,
                                             1024, nullptr, buffer.ptr()));
            ASSERT_NE(buffer, nullptr);
        }

        // mark the first set of processing as done and notify the main thread
        std::unique_lock<std::mutex> lock(m);
        thread_done = true;
        lock.unlock();
        cv.notify_one();

        // wait for the main thread to release the first context
        lock.lock();
        cv.wait(lock, [&] { return released; });

        {
            // create a queue with the 2nd context
            uur::raii::Queue queue = nullptr;
            ASSERT_SUCCESS(
                urQueueCreate(context2, device, nullptr, queue.ptr()));
            ASSERT_NE(queue, nullptr);

            // ensure queue has correct context
            ASSERT_EQ(context2, queue->getContext());

            // create a buffer to set the active context
            uur::raii::Mem buffer = nullptr;
            ASSERT_SUCCESS(urMemBufferCreate(context2, UR_MEM_FLAG_READ_WRITE,
                                             1024, nullptr, buffer.ptr()));

            // check that the 2nd context is now tha active cuda context
            ASSERT_SUCCESS_CUDA(cuCtxGetCurrent(&current));
            ASSERT_EQ(current, context2->get());
        }
    });

    // wait for the thread to be done with the first queue to release the first
    // context
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&] { return thread_done; });

    // notify the other thread that the context was released
    released = true;
    lock.unlock();
    cv.notify_one();

    // wait for the thread to finish
    test_thread.join();
}
