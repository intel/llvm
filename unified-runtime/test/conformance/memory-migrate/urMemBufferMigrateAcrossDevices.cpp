// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Some tests to ensure implicit memory migration of buffers across devices
// in the same context.

#include "uur/fixtures.h"
#include <thread>

using T = uint32_t;

struct urMultiDeviceContextTest : uur::urPlatformTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urPlatformTest::SetUp());
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
    if (num_devices <= 1) {
      return;
    }

    devices = std::vector<ur_device_handle_t>(num_devices);
    ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, num_devices,
                               devices.data(), nullptr));
    ASSERT_SUCCESS(
        urContextCreate(num_devices, devices.data(), nullptr, &context));

    queues = std::vector<ur_queue_handle_t>(num_devices);
    for (auto i = 0u; i < num_devices; ++i) {
      ASSERT_SUCCESS(urQueueCreate(context, devices[i], nullptr, &queues[i]));
    }
  }

  void TearDown() override {
    uur::urPlatformTest::TearDown();
    if (num_devices <= 1) {
      return;
    }
    for (auto i = 0u; i < num_devices; ++i) {
      urDeviceRelease(devices[i]);
      urQueueRelease(queues[i]);
    }
    urContextRelease(context);
  }

  uint32_t num_devices = 0;
  ur_context_handle_t context;
  std::vector<ur_device_handle_t> devices;
  std::vector<ur_queue_handle_t> queues;
};

struct urMultiDeviceContextMemBufferTest : urMultiDeviceContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::SetUp());
    if (num_devices <= 1) {
      return;
    }
    ASSERT_SUCCESS(urMemBufferCreate(context, 0 /*flags=*/, buffer_size_bytes,
                                     nullptr /*pProperties=*/, &buffer));

    UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
        program_name, platform, il_binary));

    programs = std::vector<ur_program_handle_t>(num_devices);
    kernels = std::vector<ur_kernel_handle_t>(num_devices);

    const ur_program_properties_t properties = {
        UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
        static_cast<uint32_t>(metadatas.size()),
        metadatas.empty() ? nullptr : metadatas.data()};
    for (auto i = 0u; i < num_devices; ++i) {
      UUR_RETURN_ON_FATAL_FAILURE(
          uur::KernelsEnvironment::instance->CreateProgram(
              platform, context, devices[i], *il_binary, &properties,
              &programs[i]));
      ASSERT_SUCCESS(urProgramBuild(context, programs[i], nullptr));
      auto kernel_names =
          uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
      kernel_name = kernel_names[0];
      ASSERT_FALSE(kernel_name.empty());
      ASSERT_SUCCESS(
          urKernelCreate(programs[i], kernel_name.data(), &kernels[i]));
    }
  }

  // Adds a kernel arg representing a sycl buffer constructed with a 1D range.
  void AddBuffer1DArg(ur_kernel_handle_t kernel, size_t current_arg_index,
                      ur_mem_handle_t buffer) {
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index, nullptr, buffer));

    // SYCL device kernels have different interfaces depending on the
    // backend being used. Typically a kernel which takes a buffer argument
    // will take a pointer to the start of the buffer and a sycl::id param
    // which is a struct that encodes the accessor to the buffer. However
    // the AMD backend handles this differently and uses three separate
    // arguments for each of the three dimensions of the accessor.

    ur_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));
    if (backend == UR_BACKEND_HIP) {
      // this emulates the three offset params for buffer accessor on AMD.
      size_t val = 0;
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 1,
                                         sizeof(size_t), nullptr, &val));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 2,
                                         sizeof(size_t), nullptr, &val));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 3,
                                         sizeof(size_t), nullptr, &val));
      current_arg_index += 4;
    } else {
      // This emulates the offset struct sycl adds for a 1D buffer accessor.
      struct {
        size_t offsets[1] = {0};
      } accessor;
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 1,
                                         sizeof(accessor), nullptr, &accessor));
      current_arg_index += 2;
    }
  }

  void TearDown() override {
    if (num_devices > 1) {
      for (auto i = 0u; i < num_devices; ++i) {
        ASSERT_SUCCESS(urKernelRelease(kernels[i]));
        ASSERT_SUCCESS(urProgramRelease(programs[i]));
      }
      urMemRelease(buffer);
    }
    urMultiDeviceContextTest::TearDown();
  }

  size_t buffer_size = 4096;
  size_t buffer_size_bytes = 4096 * sizeof(T);
  ur_mem_handle_t buffer;

  // Program stuff so we can launch kernels
  std::shared_ptr<std::vector<char>> il_binary;
  std::string program_name = "inc";
  std::string kernel_name;
  std::vector<ur_program_handle_t> programs;
  std::vector<ur_kernel_handle_t> kernels;
  std::vector<ur_program_metadata_t> metadatas{};
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMultiDeviceContextMemBufferTest);

TEST_P(urMultiDeviceContextMemBufferTest, WriteRead) {
  if (num_devices <= 1) {
    GTEST_SKIP();
  }
  T fill_val = 42;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> out_vec(buffer_size, 0);
  ur_event_handle_t e1;

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                         buffer_size_bytes, in_vec.data(), 0,
                                         nullptr, &e1));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                        buffer_size_bytes, out_vec.data(), 1,
                                        &e1, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queues[1]));

  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val);
  }
}

TEST_P(urMultiDeviceContextMemBufferTest, FillRead) {
  if (num_devices <= 1) {
    GTEST_SKIP();
  }
  T fill_val = 42;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> out_vec(buffer_size);
  ur_event_handle_t e1;

  ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &fill_val,
                                        sizeof(fill_val), 0, buffer_size_bytes,
                                        0, nullptr, &e1));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                        buffer_size_bytes, out_vec.data(), 1,
                                        &e1, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queues[1]));

  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val);
  }
}

TEST_P(urMultiDeviceContextMemBufferTest, WriteKernelRead) {
  if (num_devices <= 1) {
    GTEST_SKIP();
  }

  // Kernel to run on queues[1]
  AddBuffer1DArg(kernels[1], 0, buffer);

  T fill_val = 42;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> out_vec(buffer_size);
  ur_event_handle_t e1, e2;

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                         buffer_size_bytes, in_vec.data(), 0,
                                         nullptr, &e1));

  size_t work_dims[3] = {buffer_size, 1, 1};
  size_t offset[3] = {0, 0, 0};

  // Kernel increments the fill val by 1
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[1], kernels[1], 1 /*workDim=*/,
                                       offset, work_dims, nullptr, nullptr, 1,
                                       &e1, &e2));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[0], buffer, false, 0,
                                        buffer_size_bytes, out_vec.data(), 1,
                                        &e2, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queues[0]));

  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val + 1);
  }
}

TEST_P(urMultiDeviceContextMemBufferTest, WriteKernelKernelRead) {
  if (num_devices <= 1) {
    GTEST_SKIP();
  }

  AddBuffer1DArg(kernels[0], 0, buffer);
  AddBuffer1DArg(kernels[1], 0, buffer);

  T fill_val = 42;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> out_vec(buffer_size);
  ur_event_handle_t e1, e2, e3;

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                         buffer_size_bytes, in_vec.data(), 0,
                                         nullptr, &e1));

  size_t work_dims[3] = {buffer_size, 1, 1};
  size_t offset[3] = {0, 0, 0};

  // Kernel increments the fill val by 1
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[1], kernels[1], 1 /*workDim=*/,
                                       offset, work_dims, nullptr, nullptr, 1,
                                       &e1, &e2));

  // Kernel increments the fill val by 1
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[0], kernels[0], 1 /*workDim=*/,
                                       offset, work_dims, nullptr, nullptr, 1,
                                       &e2, &e3));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                        buffer_size_bytes, out_vec.data(), 1,
                                        &e3, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queues[1]));

  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val + 2);
  }
}

TEST_P(urMultiDeviceContextMemBufferTest, PingPongKernelExecution) {
  if (num_devices <= 1) {
    GTEST_SKIP();
  }

  // Setup kernels for alternating devices
  AddBuffer1DArg(kernels[0], 0, buffer);
  AddBuffer1DArg(kernels[1], 0, buffer);

  T fill_val = 50;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> out_vec(buffer_size);

  const uint32_t ping_pong_iterations = 20;
  std::vector<ur_event_handle_t> events(ping_pong_iterations);
  ur_event_handle_t write_event, read_event;

  size_t work_dims[3] = {buffer_size, 1, 1};
  size_t offset[3] = {0, 0, 0};

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                         buffer_size_bytes, in_vec.data(), 0,
                                         nullptr, &write_event));

  // Ping-pong kernel execution across two devices
  // Each kernel increments the buffer values by 1
  for (uint32_t i = 0; i < ping_pong_iterations; ++i) {
    uint32_t device_idx = i % 2;
    ur_event_handle_t *wait_event = (i == 0) ? &write_event : &events[i - 1];

    ASSERT_SUCCESS(urEnqueueKernelLaunch(
        queues[device_idx], kernels[device_idx], 1, offset, work_dims, nullptr,
        0, nullptr, 1, wait_event, &events[i]));
  }

  ASSERT_SUCCESS(urEnqueueMemBufferRead(
      queues[0], buffer, false, 0, buffer_size_bytes, out_vec.data(), 1,
      &events[ping_pong_iterations - 1], &read_event));

  ASSERT_SUCCESS(urEventWait(1, &read_event));

  // Verify final result
  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val + ping_pong_iterations);
  }
}

TEST_P(urMultiDeviceContextMemBufferTest, ComplexMigrationPattern) {
  if (num_devices <= 1) {
    GTEST_SKIP();
  }

  for (uint32_t i = 0; i < num_devices; ++i) {
    AddBuffer1DArg(kernels[i], 0, buffer);
  }

  T fill_val = 200;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> intermediate_vec(buffer_size);
  std::vector<T> out_vec(buffer_size);

  ur_event_handle_t write_event, intermediate_read_event,
      intermediate_write_event, final_kernel_event, read_event;

  size_t work_dims[3] = {buffer_size, 1, 1};
  size_t offset[3] = {0, 0, 0};

  // Write on device 0, execute kernel on device 1
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                         buffer_size_bytes, in_vec.data(), 0,
                                         nullptr, &write_event));

  ur_event_handle_t phase1_kernel_event;
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[1], kernels[1], 1, offset,
                                       work_dims, nullptr, 0, nullptr, 1,
                                       &write_event, &phase1_kernel_event));

  // Read intermediate result back to host
  ASSERT_SUCCESS(urEnqueueMemBufferRead(
      queues[0], buffer, false, 0, buffer_size_bytes, intermediate_vec.data(),
      1, &phase1_kernel_event, &intermediate_read_event));

  // Modify on host and write back through different device
  ASSERT_SUCCESS(urEventWait(1, &intermediate_read_event));
  for (auto &val : intermediate_vec) {
    val += 10;
  }

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(
      queues[1], buffer, false, 0, buffer_size_bytes, intermediate_vec.data(),
      0, nullptr, &intermediate_write_event));

  // Final kernel execution
  ASSERT_SUCCESS(urEnqueueKernelLaunch(
      queues[0], kernels[0], 1, offset, work_dims, nullptr, 0, nullptr, 1,
      &intermediate_write_event, &final_kernel_event));

  // Final read
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                        buffer_size_bytes, out_vec.data(), 1,
                                        &final_kernel_event, &read_event));

  ASSERT_SUCCESS(urEventWait(1, &read_event));

  // Verify result: initial + 1 (phase1) + 10 (host) + 1 (final) = initial + 12
  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val + 12);
  }
}

TEST_P(urMultiDeviceContextMemBufferTest, KernelsExecutionWithThreads) {
  AddBuffer1DArg(kernels[0], 0, buffer);

  T fill_val = 100;
  std::vector<T> in_vec(buffer_size, fill_val);
  std::vector<T> out_vec(buffer_size);

  const uint32_t thread_count = 8;
  const uint32_t iterations_per_thread = 10;
  std::vector<std::thread> threads(thread_count);
  ur_event_handle_t write_event, read_event;

  size_t work_dims[3] = {buffer_size, 1, 1};
  size_t offset[3] = {0, 0, 0};

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                         buffer_size_bytes, in_vec.data(), 0,
                                         nullptr, &write_event));

  // Each thread will perform a series of kernel launches on the same device
  for (auto t = 0u; t < thread_count; ++t) {
    threads[t] = std::thread([&, t]() {
      ur_event_handle_t last_event = write_event;
      for (auto i = 0u; i < iterations_per_thread; ++i) {
        ur_event_handle_t kernel_event;
        ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[0], kernels[0], 1, offset,
                                             work_dims, nullptr, 0, nullptr, 1,
                                             &last_event, &kernel_event));
        last_event = kernel_event;
      }
      // The last thread to finish will enqueue the read
      if (t == thread_count - 1) {
        ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[0], buffer, false, 0,
                                              buffer_size_bytes, out_vec.data(),
                                              1, &last_event, &read_event));
      }
    });
  }
  for (auto &th : threads) {
    th.join();
  }

  ASSERT_SUCCESS(urEventWait(1, &read_event));
  for (auto &a : out_vec) {
    ASSERT_EQ(a, fill_val + thread_count * iterations_per_thread);
  }
}
