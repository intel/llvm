// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <memory>
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urEnqueueKernelLaunchTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "fill";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  uint32_t val = 42;
  size_t global_size = 32;
  size_t global_offset = 0;
  size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchTest);

struct urEnqueueKernelLaunchKernelWgSizeTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "fixed_wg_size";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  std::array<size_t, 3> global_size{32, 32, 32};
  std::array<size_t, 3> global_offset{0, 0, 0};
  // This value correlates to work_group_size<8, 4, 2> in fixed_wg_size.cpp.
  // In SYCL, the right-most dimension varies the fastest in linearization.
  // In UR, this is on the left, so we reverse the order of these values.
  std::array<size_t, 3> wg_size{2, 4, 8};
  size_t n_dimensions = 3;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchKernelWgSizeTest);

// Note: Due to an issue with HIP, the subgroup test is not generated
struct urEnqueueKernelLaunchKernelSubGroupTest : uur::urKernelExecutionTest {
  void SetUp() override {
    // Subgroup size of 8 isn't supported on the Data Center GPU Max
    UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::LevelZero{"Data Center GPU Max"},
                         uur::LevelZeroV2{"Data Center GPU Max"});

    program_name = "subgroup";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  std::array<size_t, 3> global_size{32, 32, 32};
  std::array<size_t, 3> global_offset{0, 0, 0};
  size_t n_dimensions = 3;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchKernelSubGroupTest);

struct urEnqueueKernelLaunchKernelStandardTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "standard_types";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  size_t n_dimensions = 1;
  size_t global_size = 1;
  size_t offset = 0;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchKernelStandardTest);

TEST_P(urEnqueueKernelLaunchTest, Success) {
  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(val) * global_size, &buffer);
  AddPodArg(val);
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr, 0,
                                       nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer(buffer, sizeof(val) * global_size, val);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(nullptr, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullPointer) {
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions, nullptr,
                                         &global_size, nullptr, 0, nullptr,
                                         nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, nullptr, nullptr, 0,
                                         nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, nullptr, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullPtrEventWaitList) {
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         1, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t validEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, &validEvent, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         1, &inv_evt, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueKernelLaunchTest, InvalidWorkDimension) {
  uint32_t max_work_item_dimensions = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(
      device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
      sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr));
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel,
                                         max_work_item_dimensions + 1,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidWorkGroupSize) {
  // As far as I can tell, there's no way to check if a kernel or device
  // requires uniform work group sizes or not, so this may succeed or report
  // an error
  size_t local_size = 31;
  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(val) * global_size, &buffer);
  AddPodArg(val);
  auto result =
      urEnqueueKernelLaunch(queue, kernel, n_dimensions, &global_offset,
                            &global_size, &local_size, 0, nullptr, nullptr);
  ASSERT_TRUE(result == UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE ||
              result == UR_RESULT_SUCCESS);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidKernelArgs) {
  // Cuda and hip both lack any way to validate kernel args
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  ur_platform_backend_t backend;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(ur_platform_backend_t), &backend,
                                   nullptr));

  if (backend == UR_PLATFORM_BACKEND_CUDA ||
      backend == UR_PLATFORM_BACKEND_HIP ||
      backend == UR_PLATFORM_BACKEND_LEVEL_ZERO) {
    GTEST_FAIL() << "AMD, L0 and Nvidia can't check kernel arguments.";
  }

  // Enqueue kernel without setting any args
  ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_KERNEL_ARGS);
}

TEST_P(urEnqueueKernelLaunchKernelWgSizeTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       global_offset.data(), global_size.data(),
                                       nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
}

TEST_P(urEnqueueKernelLaunchKernelWgSizeTest, SuccessWithExplicitLocalSize) {
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       global_offset.data(), global_size.data(),
                                       wg_size.data(), 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
}

TEST_P(urEnqueueKernelLaunchKernelWgSizeTest, NonMatchingLocalSize) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  std::array<size_t, 3> wrong_wg_size{8, 8, 8};
  ASSERT_EQ_RESULT(
      urEnqueueKernelLaunch(queue, kernel, n_dimensions, global_offset.data(),
                            global_size.data(), wrong_wg_size.data(), 0,
                            nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
}

TEST_P(urEnqueueKernelLaunchKernelSubGroupTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(size_t), &buffer);
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       global_offset.data(), global_size.data(),
                                       nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  // We specify this subgroup size in the kernel source, and then the kernel
  // queries for its subgroup size at runtime and writes it to the buffer.
  ValidateBuffer<size_t>(buffer, sizeof(size_t), 8);
}

struct Pair {
  uint32_t a;
  uint32_t b;
};
TEST_P(urEnqueueKernelLaunchKernelStandardTest, Success) {
  uint32_t expected_result = 2410;
  ur_mem_handle_t output = nullptr;
  AddBuffer1DArg(sizeof(uint32_t), &output);
  AddPodArg(true);
  AddPodArg<uint8_t>(2);
  AddPodArg<uint32_t>(3);
  AddPodArg<uint64_t>(5);
  AddPodArg<Pair>({7, 5});
  AddPodArg<float>(11.0);

  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions, &offset,
                                       &global_size, nullptr, 0, nullptr,
                                       nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer<uint32_t>(output, sizeof(uint32_t), expected_result);
}

struct testParametersEnqueueKernel {
  size_t X, Y, Z;
  size_t Dims;
};

template <typename T>
inline std::string printKernelLaunchTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  std::stringstream test_name;
  test_name << platform_device_name << "__" << std::get<1>(info.param).Dims
            << "D_" << std::get<1>(info.param).X;
  if (std::get<1>(info.param).Dims > 1) {
    test_name << "_" << std::get<1>(info.param).Y;
  }
  if (std::get<1>(info.param).Dims > 2) {
    test_name << "_" << std::get<1>(info.param).Z;
  }
  test_name << "";
  return test_name.str();
}

struct urEnqueueKernelLaunchTestWithParam
    : uur::urKernelExecutionTestWithParam<testParametersEnqueueKernel> {
  void SetUp() override {
    global_range[0] = std::get<1>(GetParam()).X;
    global_range[1] = std::get<1>(GetParam()).Y;
    global_range[2] = std::get<1>(GetParam()).Z;
    buffer_size = sizeof(val) * global_range[0];
    n_dimensions = std::get<1>(GetParam()).Dims;
    if (n_dimensions == 1) {
      program_name = "fill";
    } else if (n_dimensions == 2) {
      program_name = "fill_2d";
      buffer_size *= global_range[1];
    } else {
      assert(n_dimensions == 3);
      program_name = "fill_3d";
      buffer_size *= global_range[1] * global_range[2];
    }
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTestWithParam::SetUp());
  }

  void TearDown() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTestWithParam<
                                testParametersEnqueueKernel>::TearDown());
  }

  uint32_t val = 42;
  size_t global_range[3];
  size_t global_offset[3] = {0, 0, 0};
  size_t n_dimensions;
  size_t buffer_size;
};

static std::vector<testParametersEnqueueKernel> test_cases{// 1D
                                                           {1, 1, 1, 1},
                                                           {31, 1, 1, 1},
                                                           {1027, 1, 1, 1},
                                                           {32, 1, 1, 1},
                                                           {256, 1, 1, 1},
                                                           // 2D
                                                           {1, 1, 1, 2},
                                                           {31, 7, 1, 2},
                                                           {1027, 1, 1, 2},
                                                           {1, 32, 1, 2},
                                                           {256, 79, 1, 2},
                                                           // 3D
                                                           {1, 1, 1, 3},
                                                           {31, 7, 1, 3},
                                                           {1027, 1, 19, 3},
                                                           {1, 53, 19, 3},
                                                           {256, 79, 8, 3}};
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueKernelLaunchTestWithParam, testing::ValuesIn(test_cases),
    printKernelLaunchTestString<urEnqueueKernelLaunchTestWithParam>);

TEST_P(urEnqueueKernelLaunchTestWithParam, Success) {
  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(buffer_size, &buffer);
  AddPodArg(val);
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       global_offset, global_range, nullptr, 0,
                                       nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer(buffer, buffer_size, val);
}

struct urEnqueueKernelLaunchWithUSM : uur::urKernelExecutionTest {

  void SetUp() override {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

    ur_device_usm_access_capability_flags_t device_usm = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
    if (!device_usm) {
      GTEST_SKIP() << "Device USM is not supported";
    }

    alloc_size = 1024;

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    alloc_size, &usmPtr));

    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  void TearDown() override {

    if (usmPtr) {
      EXPECT_SUCCESS(urUSMFree(context, usmPtr));
    }

    UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::TearDown());
  }

  size_t alloc_size = 0;
  void *usmPtr = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchWithUSM);

TEST_P(urEnqueueKernelLaunchWithUSM, Success) {
  size_t work_dim = 1;
  size_t global_offset = 0;
  size_t global_size = alloc_size / sizeof(uint32_t);
  uint32_t fill_val = 42;

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, usmPtr));
  ASSERT_SUCCESS(
      urKernelSetArgValue(kernel, 1, sizeof(fill_val), nullptr, &fill_val));

  auto *ptr = static_cast<uint32_t *>(usmPtr);
  for (size_t i = 0; i < global_size; i++) {
    ptr[i] = 0;
  }

  ur_event_handle_t kernel_evt;
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, work_dim, &global_offset,
                                       &global_size, nullptr, 0, nullptr,
                                       &kernel_evt));

  ASSERT_SUCCESS(urQueueFinish(queue));

  // verify fill worked
  for (size_t i = 0; i < global_size; i++) {
    ASSERT_EQ(ptr[i], fill_val);
  }
}

TEST_P(urEnqueueKernelLaunchWithUSM, WithMemcpy) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});

  size_t work_dim = 1;
  size_t global_offset = 0;
  size_t global_size = alloc_size / sizeof(uint32_t);
  uint32_t fill_val = 42;

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, usmPtr));
  ASSERT_SUCCESS(
      urKernelSetArgValue(kernel, 1, sizeof(fill_val), nullptr, &fill_val));

  std::vector<uint32_t> input(global_size, 0);
  std::vector<uint32_t> data(global_size);

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, usmPtr, input.data(),
                                    alloc_size, 0, nullptr, nullptr));

  ur_event_handle_t kernel_evt;
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, work_dim, &global_offset,
                                       &global_size, nullptr, 0, nullptr,
                                       &kernel_evt));

  ur_event_handle_t memcpy_event;
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, data.data(), usmPtr,
                                    alloc_size, 1, &kernel_evt, &memcpy_event));

  ASSERT_SUCCESS(urEventWait(1, &memcpy_event));

  // verify fill worked
  for (size_t i = 0; i < global_size; i++) {
    ASSERT_EQ(data[i], fill_val);
  }
}

struct urEnqueueKernelLaunchWithVirtualMemory : uur::urKernelExecutionTest {

  void SetUp() override {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

    ur_bool_t virtual_memory_support = false;
    ASSERT_SUCCESS(
        urDeviceGetInfo(device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT,
                        sizeof(ur_bool_t), &virtual_memory_support, nullptr));
    if (!virtual_memory_support) {
      GTEST_SKIP() << "Virtual memory is not supported.";
    }

    ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(
        context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
        sizeof(granularity), &granularity, nullptr));

    alloc_size = 1024;
    virtual_page_size = uur::RoundUpToNearestFactor(alloc_size, granularity);

    ASSERT_SUCCESS(urPhysicalMemCreate(context, device, virtual_page_size,
                                       nullptr, &physical_mem));

    ASSERT_SUCCESS(
        urVirtualMemReserve(context, nullptr, virtual_page_size, &virtual_ptr));

    ASSERT_SUCCESS(urVirtualMemMap(context, virtual_ptr, virtual_page_size,
                                   physical_mem, 0,
                                   UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE));

    int pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, virtual_ptr, sizeof(pattern),
                                    &pattern, virtual_page_size, 0, nullptr,
                                    nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  void TearDown() override {

    if (virtual_ptr) {
      EXPECT_SUCCESS(
          urVirtualMemUnmap(context, virtual_ptr, virtual_page_size));
      EXPECT_SUCCESS(urVirtualMemFree(context, virtual_ptr, virtual_page_size));
    }

    if (physical_mem) {
      EXPECT_SUCCESS(urPhysicalMemRelease(physical_mem));
    }

    UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::TearDown());
  }

  size_t granularity = 0;
  size_t alloc_size = 0;
  size_t virtual_page_size = 0;
  ur_physical_mem_handle_t physical_mem = nullptr;
  void *virtual_ptr = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchWithVirtualMemory);

TEST_P(urEnqueueKernelLaunchWithVirtualMemory, Success) {
  size_t work_dim = 1;
  size_t global_offset = 0;
  size_t global_size = alloc_size / sizeof(uint32_t);
  uint32_t fill_val = 42;

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, virtual_ptr));
  ASSERT_SUCCESS(
      urKernelSetArgValue(kernel, 1, sizeof(fill_val), nullptr, &fill_val));

  ur_event_handle_t kernel_evt;
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, work_dim, &global_offset,
                                       &global_size, nullptr, 0, nullptr,
                                       &kernel_evt));

  std::vector<uint32_t> data(global_size);
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, data.data(), virtual_ptr,
                                    alloc_size, 1, &kernel_evt, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queue));

  // verify fill worked
  for (size_t i = 0; i < data.size(); i++) {
    ASSERT_EQ(fill_val, data.at(i));
  }
}

struct urEnqueueKernelLaunchMultiDeviceTest
    : public uur::urMultiDeviceContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiDeviceContextTest::SetUp());
    queues.reserve(uur::DevicesEnvironment::instance->devices.size());
    for (const auto &device : devices) {
      ur_queue_handle_t queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(this->context, device, 0, &queue));
      queues.push_back(queue);
    }
    auto kernelName =
        uur::KernelsEnvironment::instance->GetEntryPointNames("foo")[0];

    uur::KernelsEnvironment::instance->LoadSource("foo", platform, il_binary);

    ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
        platform, context, devices[0], *il_binary, nullptr, &program));

    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
    ASSERT_SUCCESS(urKernelCreate(program, kernelName.data(), &kernel));
  }

  void TearDown() override {
    for (const auto &queue : queues) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    if (program) {
      EXPECT_SUCCESS(urProgramRelease(program));
    }
    if (kernel) {
      EXPECT_SUCCESS(urKernelRelease(kernel));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiDeviceContextTest::TearDown());
  }

  ur_program_handle_t program = nullptr;
  ur_kernel_handle_t kernel = nullptr;

  std::shared_ptr<std::vector<char>> il_binary;
  std::vector<ur_queue_handle_t> queues;

  uint32_t val = 42;
  size_t global_size = 32;
  size_t global_offset = 0;
  size_t n_dimensions = 1;
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urEnqueueKernelLaunchMultiDeviceTest);

// TODO: rewrite this test, right now it only works for a single queue
// (the context is only created for one device)
TEST_P(urEnqueueKernelLaunchMultiDeviceTest, KernelLaunchReadDifferentQueues) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::LevelZero{}, uur::LevelZeroV2{});

  uur::KernelLaunchHelper helper =
      uur::KernelLaunchHelper{platform, context, kernel, queues[0]};

  ur_mem_handle_t buffer = nullptr;
  helper.AddBuffer1DArg(sizeof(val) * global_size, &buffer, nullptr);
  helper.AddPodArg(val);
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[0], kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr, 0,
                                       nullptr, nullptr));

  // Wait for the queue to finish executing.
  EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

  // Then the remaining queues do blocking reads from the buffer. Since the
  // queues target different devices this checks that any devices memory has
  // been synchronized.
  for (unsigned i = 1; i < queues.size(); ++i) {
    const auto queue = queues[i];
    uint32_t output = 0;
    ASSERT_SUCCESS(urEnqueueMemBufferRead(
        queue, buffer, true, 0, sizeof(output), &output, 0, nullptr, nullptr));
    ASSERT_EQ(val, output) << "Result on queue " << i << " did not match!";
  }
}

struct urEnqueueKernelLaunchUSMLinkedList
    : uur::urKernelTestWithParam<uur::BoolTestParam> {
  struct Node {
    Node() : next(nullptr), num(0xDEADBEEF) {}

    Node *next;
    uint32_t num;
  };

  void SetUp() override {
    program_name = "usm_ll";
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urKernelTestWithParam<uur::BoolTestParam>::SetUp());

    use_pool = getParam().value;
    ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
    ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr, 0};
    if (use_pool) {
      ur_bool_t poolSupport = false;
      ASSERT_SUCCESS(uur::GetDeviceUSMPoolSupport(device, poolSupport));
      if (!poolSupport) {
        GTEST_SKIP() << "USM pools are not supported.";
      }
      ASSERT_SUCCESS(urUSMPoolCreate(this->context, &pool_desc, &pool));
    }
  }

  void TearDown() override {
    auto *list_cur = list_head;
    while (list_cur) {
      auto *list_next = list_cur->next;
      ASSERT_SUCCESS(urUSMFree(context, list_cur));
      list_cur = list_next;
    }

    if (queue) {
      ASSERT_SUCCESS(urQueueRelease(queue));
    }

    if (pool) {
      ASSERT_SUCCESS(urUSMPoolRelease(pool));
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urKernelTestWithParam<uur::BoolTestParam>::TearDown());
  }

  size_t global_size = 1;
  size_t global_offset = 0;
  Node *list_head = nullptr;
  const int num_nodes = 4;
  bool use_pool = false;
  ur_usm_pool_handle_t pool = nullptr;
  ur_queue_handle_t queue = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueKernelLaunchUSMLinkedList,
    testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
    uur::deviceTestWithParamPrinter<uur::BoolTestParam>);

TEST_P(urEnqueueKernelLaunchUSMLinkedList, Success) {
  if (use_pool) {
    UUR_KNOWN_FAILURE_ON(uur::HIP{});
  }

  ur_device_usm_access_capability_flags_t shared_usm_flags = 0;
  ASSERT_SUCCESS(
      uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
  if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Shared USM is not supported.";
  }

  // Build linked list with USM allocations
  ur_usm_desc_t desc{UR_STRUCTURE_TYPE_USM_DESC, nullptr, 0, alignof(Node)};
  ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &desc, pool, sizeof(Node),
                                  reinterpret_cast<void **>(&list_head)));
  ASSERT_NE(list_head, nullptr);
  Node *list_cur = list_head;
  for (int i = 0; i < num_nodes; i++) {
    list_cur->num = i * 2;
    if (i < num_nodes - 1) {
      ASSERT_SUCCESS(
          urUSMSharedAlloc(context, device, &desc, pool, sizeof(Node),
                           reinterpret_cast<void **>(&list_cur->next)));
      ASSERT_NE(list_cur->next, nullptr);
    } else {
      list_cur->next = nullptr;
    }
    list_cur = list_cur->next;
  }

  ur_bool_t indirect = true;
  ASSERT_SUCCESS(urKernelSetExecInfo(kernel,
                                     UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                                     sizeof(indirect), nullptr, &indirect));

  // Run kernel which will iterate the list and modify the values
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, list_head));
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, 1, &global_offset,
                                       &global_size, nullptr, 0, nullptr,
                                       nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify values
  list_cur = list_head;
  for (int i = 0; i < num_nodes; i++) {
    ASSERT_EQ(list_cur->num, i * 4 + 1);
    list_cur = list_cur->next;
  }
}
